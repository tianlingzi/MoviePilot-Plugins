import time
import random
import hashlib
import requests


class BaiduTranslate:
    _appid: str = None
    _secret_key: str = None
    _api_url: str = "https://fanyi-api.baidu.com/api/trans/vip/translate"

    def __init__(self, appid: str = None, secret_key: str = None):
        self._appid = appid
        self._secret_key = secret_key

    def _generate_sign(self, query: str, salt: str) -> str:
        """
        生成签名
        :param query: 待翻译文本
        :param salt: 随机数
        :return: 签名
        """
        sign_str = f"{self._appid}{query}{salt}{self._secret_key}"
        return hashlib.md5(sign_str.encode()).hexdigest()

    def translate_to_zh(self, text: str, context: str = None, max_retries: int = 3):
        """
        翻译为中文
        :param text: 输入文本
        :param context: 翻译上下文（百度翻译API不使用上下文，保留参数以保持接口兼容）
        :param max_retries: 最大重试次数
        :return: (是否成功, 翻译结果或错误信息)
        """
        if not self._appid or not self._secret_key:
            return False, "未配置百度翻译appid或secret_key"

        last_error = ""
        for attempt in range(max_retries + 1):
            try:
                salt = str(int(time.time()))
                sign = self._generate_sign(text, salt)

                params = {
                    'q': text,
                    'from': 'auto',
                    'to': 'zh',
                    'appid': self._appid,
                    'salt': salt,
                    'sign': sign
                }

                response = requests.post(self._api_url, data=params, timeout=30)
                result = response.json()

                if 'error_code' in result:
                    error_code = result['error_code']
                    error_msg = result.get('error_msg', '未知错误')
                    last_error = f"错误码 {error_code}: {error_msg}"
                    
                    if error_code == '54001':
                        return False, "签名错误，请检查appid和secret_key"
                    elif error_code == '54003':
                        return False, "访问频率受限，请稍后重试"
                    elif error_code == '54004':
                        return False, "账户余额不足"
                    elif error_code == '58002':
                        return False, "服务未开通，请在百度翻译控制台开通服务"
                else:
                    translations = result.get('trans_result', [])
                    if translations:
                        translated_text = '\n'.join([item.get('dst', '') for item in translations])
                        return True, translated_text.strip()
                    else:
                        last_error = "未获取到翻译结果"

            except Exception as e:
                last_error = str(e)

            if attempt < max_retries:
                base_delay = 2 ** attempt
                jitter = random.uniform(0.1, 0.9)
                sleep_time = base_delay + jitter
                print(f"翻译请求失败 (第{attempt + 1}次尝试)：{last_error}，{sleep_time:.1f}秒后重试...")
                time.sleep(sleep_time)
            else:
                print(f"翻译请求失败 (已重试{max_retries}次)：{last_error}")
                return False, last_error

        return False, last_error