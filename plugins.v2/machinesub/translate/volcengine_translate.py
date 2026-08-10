import time
import random
import hashlib
import hmac
import json
import requests
from datetime import datetime, timezone
from urllib.parse import quote
from functools import reduce
from app.log import logger


def _hmac_sha256(key, content):
    return hmac.new(key, content.encode('utf-8'), hashlib.sha256).digest()


def _sha256(content):
    if isinstance(content, str):
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    else:
        return hashlib.sha256(content).hexdigest()


def _to_hex(content):
    lst = []
    for ch in content:
        hv = hex(ch).replace('0x', '')
        if len(hv) == 1:
            hv = '0' + hv
        lst.append(hv)
    return reduce(lambda x, y: x + y, lst)


def _norm_uri(path):
    return quote(path).replace('%2F', '/').replace('+', '%20')


def _norm_query(params):
    query = ''
    for key in sorted(params.keys()):
        if type(params[key]) == list:
            for k in params[key]:
                query = query + quote(key, safe='-_.~') + '=' + quote(k, safe='-_.~') + '&'
        else:
            query = query + quote(key, safe='-_.~') + '=' + quote(str(params[key]), safe='-_.~') + '&'
    query = query[:-1]
    return query.replace('+', '%20')


def _get_signing_secret_key_v4(sk, date, region, service):
    kdate = _hmac_sha256(sk.encode('utf-8'), date)
    kregion = _hmac_sha256(kdate, region)
    kservice = _hmac_sha256(kregion, service)
    return _hmac_sha256(kservice, 'request')


class VolcengineTranslate:
    def __init__(self, access_key: str = None, secret_key: str = None):
        # 二次防御：去除首尾空白（复制粘贴常见问题）
        self._access_key = access_key.strip() if access_key else None
        self._secret_key = secret_key.strip() if secret_key else None
        self._region = "cn-north-1"
        self._service = "translate"
        self._version = "2020-06-01"
        self._host = "translate.volcengineapi.com"

    def _translate_one(self, text):
        method = "POST"
        action = "TranslateText"

        t = datetime.now(timezone.utc)
        amz_date = t.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = t.strftime("%Y%m%d")

        payload = {
            "TargetLanguage": "zh",
            "TextList": [text]
        }
        body = json.dumps(payload, separators=(',', ':'))

        query_params = {
            "Action": action,
            "Version": self._version
        }

        # 关键修复：Content-Type 必须在计算签名之前就放入 headers 中，
        # 否则 signed_headers_dict 不包含 content-type，
        # 但实际请求发送时会带 Content-Type，服务端重算 canonical request 时
        # 会包含 content-type，导致签名不匹配（SignatureDoesNotMatch）。
        headers = {
            "Host": self._host,
            "X-Date": amz_date,
            "Content-Type": "application/json",
        }

        body_hash = _sha256(body)
        headers['X-Content-Sha256'] = body_hash

        signed_headers_dict = dict()
        for key in headers:
            if key in ['Content-Type', 'Content-Md5', 'Host'] or key.startswith('X-'):
                signed_headers_dict[key.lower()] = headers[key]

        if 'host' in signed_headers_dict:
            v = signed_headers_dict['host']
            if v.find(':') != -1:
                split = v.split(':')
                port = split[1]
                if str(port) == '80' or str(port) == '443':
                    signed_headers_dict['host'] = split[0]

        signed_str = ''
        for key in sorted(signed_headers_dict.keys()):
            signed_str += key + ':' + signed_headers_dict[key] + '\n'

        signed_headers_str = ';'.join(sorted(signed_headers_dict.keys()))

        canonical_uri = '/'
        canonical_querystring = _norm_query(query_params)
        canonical_request = '\n'.join([
            method,
            _norm_uri(canonical_uri),
            canonical_querystring,
            signed_str,
            signed_headers_str,
            body_hash
        ])
        hashed_canonical_request = _sha256(canonical_request)

        algorithm = "HMAC-SHA256"
        credential_scope = '/'.join([date_stamp, self._region, self._service, 'request'])
        signing_str = '\n'.join([algorithm, amz_date, credential_scope, hashed_canonical_request])

        signing_key = _get_signing_secret_key_v4(self._secret_key, date_stamp, self._region, self._service)
        sign = _to_hex(_hmac_sha256(signing_key, signing_str))

        authorization_header = f"{algorithm} Credential={self._access_key}/{credential_scope}, SignedHeaders={signed_headers_str}, Signature={sign}"
        headers['Authorization'] = authorization_header

        logger.debug(
            "[Volcengine] 请求签名详情：host=%s, action=%s, signed_headers=%s, credential_scope=%s, body_hash=%s, text_len=%s",
            self._host, action, signed_headers_str, credential_scope, body_hash, len(text)
        )

        url = f"https://{self._host}/"
        response = requests.post(url, headers=headers, params=query_params, data=body, timeout=30)

        if response.status_code == 200:
            result = response.json()
            error = result.get('ResponseMetadata', {}).get('Error')
            if error:
                # 签名失败时补充上下文，便于排查
                err_code = error.get('Code')
                err_msg = error.get('Message')
                if err_code == 'SignatureDoesNotMatch':
                    logger.error(
                        "[Volcengine] 签名不匹配。请检查：1) AK/SK 是否正确（是否有多余空格）；"
                        "2) 本地系统时间是否与 UTC 一致；3) SecretKey 是否为火山引擎控制台的 SK。"
                        "当前 AK 前4位=%s...，Date=%s",
                        (self._access_key[:4] if self._access_key else ''),
                        amz_date
                    )
                return False, f"API错误 {err_code}: {err_msg}"
            translations = result.get('TranslationList', [])
            if translations:
                translated_text = translations[0].get('Translation', '')
                if translated_text:
                    return True, translated_text.strip()
            return False, "返回体中没有 TranslationList"
        # 非 200 时补充响应文本
        try:
            raw_text = response.text[:300]
        except Exception:
            raw_text = ''
        return False, f"请求失败: HTTP {response.status_code} {raw_text}"

    def translate_to_zh(self, text: str, context: str = None, max_retries: int = 3):
        """
        翻译为中文
        :param text: 输入文本
        :param context: 翻译上下文（火山引擎翻译API不使用上下文，保留参数以保持接口兼容）
        :param max_retries: 最大重试次数
        :return: (是否成功, 翻译结果或错误信息)
        """
        if not self._access_key or not self._secret_key:
            return False, "未配置火山引擎access_key或secret_key"

        last_error = ""
        for attempt in range(max_retries + 1):
            try:
                success, result = self._translate_one(text)
                if success:
                    return True, result
                else:
                    last_error = result
            except Exception as e:
                last_error = str(e)
                logger.exception("火山引擎翻译异常")

            if attempt < max_retries:
                base_delay = 2 ** attempt
                jitter = random.uniform(0.1, 0.9)
                sleep_time = base_delay + jitter
                logger.warning(f"火山引擎翻译请求失败 (第{attempt + 1}次尝试)：{last_error}，{sleep_time:.1f}秒后重试...")
                time.sleep(sleep_time)
            else:
                logger.error(f"火山引擎翻译请求失败 (已重试{max_retries}次)：{last_error}")
                return False, last_error

        return False, last_error
