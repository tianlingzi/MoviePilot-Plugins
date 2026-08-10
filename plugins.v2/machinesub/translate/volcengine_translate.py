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


def _hmac_sha256(key, msg):
    """HMAC-SHA256，key 可为 bytes 或 str，msg 为 str。返回 bytes。"""
    if isinstance(key, str):
        key = key.encode('utf-8')
    return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()


def _sha256_hex(content):
    """SHA256 十六进制摘要。"""
    if isinstance(content, str):
        content = content.encode('utf-8')
    return hashlib.sha256(content).hexdigest()


def _norm_query(params):
    """规范化查询字符串（按 key 升序，RFC3986 编码）。"""
    query = ''
    for key in sorted(params.keys()):
        if type(params[key]) == list:
            for k in params[key]:
                query = query + quote(key, safe='-_.~') + '=' + quote(k, safe='-_.~') + '&'
        else:
            query = query + quote(key, safe='-_.~') + '=' + quote(str(params[key]), safe='-_.~') + '&'
    return query[:-1].replace('+', '%20')


def _get_signing_key(sk, date_stamp, region, service):
    """派生签名密钥（已用官方示例值验证正确，使用 raw bytes 作为下一步 key）。"""
    kdate = _hmac_sha256(sk, date_stamp)
    kregion = _hmac_sha256(kdate, region)
    kservice = _hmac_sha256(kregion, service)
    return _hmac_sha256(kservice, 'request')


class VolcengineTranslate:
    def __init__(self, access_key: str = None, secret_key: str = None):
        # 去除首尾空白（复制粘贴常见问题）
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
        body_hash = _sha256_hex(body)

        query_params = {
            "Action": action,
            "Version": self._version
        }
        canonical_querystring = _norm_query(query_params)

        # 按官方文档：SignedHeaders 只签 host;x-date
        # 不签 content-type，不签 x-content-sha256，也不发送 X-Content-Sha256 头
        canonical_headers = f"host:{self._host}\nx-date:{amz_date}\n"
        signed_headers = "host;x-date"

        canonical_request = '\n'.join([
            method,
            '/',
            canonical_querystring,
            canonical_headers,
            signed_headers,
            body_hash
        ])

        credential_scope = '/'.join([date_stamp, self._region, self._service, 'request'])
        string_to_sign = '\n'.join([
            "HMAC-SHA256",
            amz_date,
            credential_scope,
            _sha256_hex(canonical_request)
        ])

        signing_key = _get_signing_key(self._secret_key, date_stamp, self._region, self._service)
        signature = _hmac_sha256(signing_key, string_to_sign).hex()

        authorization = (
            f"HMAC-SHA256 Credential={self._access_key}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )

        # 实际发送的 headers：Content-Type 要发送但不参与签名
        headers = {
            "Host": self._host,
            "Content-Type": "application/json",
            "X-Date": amz_date,
            "Authorization": authorization,
        }

        url = f"https://{self._host}/"
        response = requests.post(url, headers=headers, params=query_params, data=body, timeout=30)

        if response.status_code == 200:
            result = response.json()
            error = result.get('ResponseMetadata', {}).get('Error')
            if error:
                err_code = error.get('Code')
                err_msg = error.get('Message')
                if err_code == 'SignatureDoesNotMatch':
                    logger.error(
                        "[Volcengine] 签名不匹配。请检查：1) AK/SK 是否正确；"
                        "2) 本地系统时间是否与 UTC 一致；3) SecretKey 是否为火山引擎控制台的 SK。"
                        "AK 前4位=%s..., Date=%s",
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
