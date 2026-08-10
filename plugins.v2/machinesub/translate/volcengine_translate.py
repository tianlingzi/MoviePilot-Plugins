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
        self._access_key = access_key
        self._secret_key = secret_key
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

        headers = {
            "Host": self._host,
            "X-Date": amz_date
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
        headers['Content-Type'] = 'application/json'

        url = f"https://{self._host}/"
        response = requests.post(url, headers=headers, params=query_params, data=body, timeout=30)

        if response.status_code == 200:
            result = response.json()
            error = result.get('ResponseMetadata', {}).get('Error')
            if error:
                return False, f"API错误 {error.get('Code')}: {error.get('Message')}"
            translations = result.get('TranslationList', [])
            if translations:
                translated_text = translations[0].get('Translation', '')
                if translated_text:
                    return True, translated_text.strip()
        return False, f"请求失败: {response.status_code}"

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
