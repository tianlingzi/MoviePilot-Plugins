import copy
import os
import tempfile
import time
import traceback
from datetime import timedelta, datetime
from pathlib import Path
from typing import Tuple, Dict, Any, List
from threading import Event
import iso639
import psutil
import srt
from lxml import etree
from dataclasses import dataclass
from enum import Enum
import queue
import threading
from uuid import uuid4
from app.core.config import settings
from app.core.context import MediaInfo
from app.core.event import eventmanager, Event as MPEvent
from app.schemas import TransferInfo
from app.schemas.types import NotificationType, EventType
from app.log import logger
from app.plugins import _PluginBase
from app.utils.system import SystemUtils
from .ffmpeg import Ffmpeg
from .translate.baidu_translate import BaiduTranslate
from .translate.volcengine_translate import VolcengineTranslate


class UserInterruptException(Exception):
    """用户中断当前任务的异常"""
    pass


class TaskSource(Enum):
    MANUAL = "manual"
    EVENT = "event"


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    IGNORED = "ignored"
    FAILED = "failed"


@dataclass
class TaskItem:
    task_id: str
    video_file: str
    source: TaskSource
    add_time: datetime
    status: TaskStatus = TaskStatus.PENDING
    complete_time: datetime = None
    force_translate: bool = False


class MachineSub(_PluginBase):
    # 插件名称
    plugin_name = "字幕生成与机器翻译"
    # 插件描述
    plugin_desc = "使用 faster-whisper 自动生成视频字幕，并支持百度翻译、火山引擎翻译将英文字幕等翻译成中文；支持强制翻译功能，忽略已有字幕检查直接处理。"
    # 插件图标
    plugin_icon = "agentresourceofficer.png"
    # 主题色
    plugin_color = "#87CEEB"
    # 插件版本
    plugin_version = "1.6"
    # 插件作者
    plugin_author = "tianlingzi"
    # 作者主页
    author_url = "https://www.tianlingzi.top/archives/264/"
    # 插件配置项ID前缀
    plugin_config_prefix = "machinesub"
    # 加载顺序
    plugin_order = 20
    # 可使用的用户级别
    auth_level = 2

    # 私有属性
    _tasks: Dict[str, TaskItem] = None
    _task_queue = None
    _consumer_thread = None
    _current_processing_task = None
    _running = False
    _event = Event()
    _enabled = None
    _clear_history = None
    _listen_transfer_event = None
    _send_notify = None
    _translate_preference = None
    _run_now = None
    _force_translate = None
    _path_list = None
    _file_size = None
    _translate_zh = None
    _translate_service = None
    _baidu_translate = None
    _volcengine_translate = None
    _enable_batch = None
    _batch_size = None
    _context_window = None
    _max_retries = None
    _enable_merge = None
    _subtitle_mode = None
    _enable_asr = None
    _auto_detect_language = None
    _huggingface_proxy = None
    _faster_whisper_model_path = None
    _faster_whisper_model = None
    _stats = None

    # 锁对象，保护队列去重检查
    _lock = threading.Lock()

    def init_plugin(self, config=None):
        # 如果没有配置信息， 则不处理
        if not config:
            return
        # 清理插件启动前的残留临时文件
        tempdir = tempfile.gettempdir()
        for file in os.listdir(tempdir):
            if file.startswith('machinesub-'):
                try:
                    os.remove(os.path.join(tempdir, file))
                    logger.info(f"清理残留临时文件：{file}")
                except Exception:
                    pass
        self._tasks = self.load_tasks()
        self._enabled = config.get('enabled', False)
        self._clear_history = config.get('clear_history', False)
        self._listen_transfer_event = config.get('listen_transfer_event', True)
        self._run_now = config.get('run_now')
        # 强制翻译仅在手动执行一次时可用
        self._force_translate = config.get('force_translate', False) if self._run_now else False
        if self._run_now:
            self._path_list = list(set(config.get('path_list').split('\n')))
        self._send_notify = config.get('send_notify', False)
        self._file_size = int(config.get('file_size')) if config.get('file_size') else 10
        # 字幕生成设置
        self._translate_preference = config.get('translate_preference', 'english_first')
        self._enable_asr = config.get('enable_asr', True)
        self._faster_whisper_model = config.get('faster_whisper_model', 'base')
        self._faster_whisper_model_path = config.get('faster_whisper_model_path',
                                                     self.get_data_path() / "faster-whisper-models")
        self._huggingface_proxy = config.get('proxy', True)
        self._auto_detect_language = config.get('auto_detect_language', False)

        self._translate_zh = config.get('translate_zh', False)
        self._translate_service = config.get('translate_service', 'baidu')
        self._enable_batch = config.get('enable_batch', True)
        self._batch_size = int(config.get('batch_size')) if config.get('batch_size') else 10
        self._context_window = int(config.get('context_window')) if config.get('context_window') else 0
        self._max_retries = int(config.get('max_retries')) if config.get('max_retries') else 3
        self._enable_merge = config.get('enable_merge', False)
        # 字幕输出模式：
        # single    = 单字幕：分别保留原声字幕（带标签）和纯中文字幕两个文件
        # bilingual = 双语字幕：仅保留中文+原文合并的一个双语字幕，删除机器生成的原声字幕
        # both      = 均保存：原声字幕 + 纯中文字幕 + 双语字幕全部保留
        self._subtitle_mode = config.get('subtitle_mode', 'bilingual')
        if self._subtitle_mode not in ('single', 'bilingual', 'both'):
            self._subtitle_mode = 'bilingual'

        if self._translate_zh:
            if self._translate_service == 'baidu':
                baidu_appid = config.get('baidu_appid')
                baidu_secret_key = config.get('baidu_secret_key')
                if baidu_appid:
                    baidu_appid = baidu_appid.strip()
                if baidu_secret_key:
                    baidu_secret_key = baidu_secret_key.strip()
                if not baidu_appid or not baidu_secret_key:
                    logger.error(f"翻译依赖于百度翻译，请先配置appid和secret_key")
                    return
                self._baidu_translate = BaiduTranslate(appid=baidu_appid, secret_key=baidu_secret_key)
            elif self._translate_service == 'volcengine':
                volc_access_key = config.get('volc_access_key')
                volc_secret_key = config.get('volc_secret_key')
                if volc_access_key:
                    volc_access_key = volc_access_key.strip()
                if volc_secret_key:
                    volc_secret_key = volc_secret_key.strip()
                if not volc_access_key or not volc_secret_key:
                    logger.error(f"翻译依赖于火山引擎翻译，请先配置access_key和secret_key")
                    return
                self._volcengine_translate = VolcengineTranslate(
                    access_key=volc_access_key,
                    secret_key=volc_secret_key
                )

        if self._clear_history:
            config['clear_history'] = False
            self.update_config(config)
            self.clear_tasks()
        if self._enabled:
            logger.info("机器字幕生成与翻译服务已启动")
            # asr 配置检查
            if self._enable_asr and not self.__check_asr():
                return

            if not self._running:
                self._task_queue = queue.Queue()
                self._consumer_thread = threading.Thread(target=self._consume_tasks, daemon=True)
                self._consumer_thread.start()
                logger.info("任务队列和消费者线程已启动")
                self._running = True

            if self._run_now:
                config['run_now'] = False
                config['force_translate'] = False
                self.update_config(config)
                if self._force_translate:
                    logger.info("立即运行一次（强制翻译模式：将忽略已有字幕检查）")
                else:
                    logger.info("立即运行一次")
                self._run_at_once(path_list=self._path_list, force_translate=self._force_translate)
        else:
            self.stop_service()

    def load_tasks(self) -> Dict[str, TaskItem]:
        raw_tasks = self.get_data("tasks") or {}
        tasks = {}
        for task_id, task_dict in raw_tasks.items():
            try:
                task = TaskItem(
                    task_id=task_dict["task_id"],
                    video_file=task_dict["video_file"],
                    source=TaskSource(task_dict["source"]),
                    add_time=datetime.fromisoformat(task_dict["add_time"]),
                    status=TaskStatus(task_dict["status"]),
                    complete_time=datetime.fromisoformat(task_dict["complete_time"])
                    if task_dict.get("complete_time") else None,
                    force_translate=task_dict.get("force_translate", False),
                )
                tasks[task_id] = task
            except Exception as e:
                logger.error(f"恢复任务失败：{e}")
        return tasks

    @staticmethod
    def _serialize_task(task: TaskItem) -> dict:
        return {
            "task_id": task.task_id,
            "video_file": task.video_file,
            "source": task.source.value,
            "add_time": task.add_time.isoformat() if task.add_time else None,
            "status": task.status.value,
            "complete_time": task.complete_time.isoformat() if task.complete_time else None,
            "force_translate": task.force_translate,
        }

    def save_tasks(self):
        tasks_dict = {task_id: self._serialize_task(task) for task_id, task in self._tasks.items()}
        self.save_data("tasks", tasks_dict)

    def add_task(self, video_file: str, source: TaskSource, force_translate: bool = False):
        """
        添加新任务到队列和任务列表中，若任务已存在则跳过。
        :param video_file: 视频文件路径
        :param source: 任务来源（手动/事件）
        :param force_translate: 是否强制翻译（忽略已有字幕检查）
        """
        task = TaskItem(
            task_id=str(uuid4()),
            video_file=video_file,
            source=source,
            add_time=datetime.now(),
            force_translate=force_translate,
        )

        if self.__is_duplicate_task(task.video_file):
            logger.info(f"任务已存在，跳过添加：{video_file}")
            return False

        self._task_queue.put(task)
        self._tasks[task.task_id] = task
        self.save_tasks()
        logger.info(f"加入任务队列: {video_file}" + ("（强制翻译）" if force_translate else ""))
        return True

    def clear_tasks(self):
        self._tasks = {task_id: task for task_id, task in self._tasks.items() if task.status in [
            TaskStatus.PENDING, TaskStatus.IN_PROGRESS
        ]}
        self.save_tasks()
        logger.info("插件历史任务已清除")

    def __is_duplicate_task(self, video_file: str) -> bool:
        with self._lock:
            for task in list(self._task_queue.queue):
                if task.video_file == video_file:
                    return True
            # 还要检查当前正在处理的任务（即可能不在队列中，但正在被消费）
            if self._consumer_thread and self._current_processing_task and self._current_processing_task.video_file == video_file:
                return True
        return False

    def _consume_tasks(self):
        while not self._event.is_set():
            try:
                task = self._task_queue.get(timeout=1)
                if task is None:
                    continue
                self._current_processing_task = task
                logger.info(f"开始处理任务 {task.task_id}: {task.video_file}")
                task.status = TaskStatus.IN_PROGRESS
                self._tasks[task.task_id] = task
                self.save_tasks()
                task.status = self.__process_autosub(task.video_file, force_translate=task.force_translate)
                task.complete_time = datetime.now()
                self._tasks[task.task_id] = task
                self.save_tasks()
                self._task_queue.task_done()
                self._current_processing_task = None
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"消费任务时发生异常: {e}")
                logger.error(traceback.format_exc())
                self._current_processing_task = None
        logger.info("消费线程已退出")

    # 监听媒体入库事件，每个事件触发一次自动字幕任务
    @eventmanager.register(EventType.TransferComplete)
    def on_transfer_complete(self, event: MPEvent):
        """监听媒体入库事件"""
        if not self._listen_transfer_event:
            return
        item = event.event_data
        item_media: MediaInfo = item.get("mediainfo")
        logger.info(f"监听到媒体入库事件：{item_media.title}")
        origin_lang = item_media.original_language
        prefer_langs = ['zh', 'chi', 'zh-CN', 'chs', 'zhs', 'zh-Hans', 'zhong', 'simp', 'cn']
        if origin_lang in prefer_langs:
            logger.info(f"媒体原始语言为中文，跳过处理")
            return

        item_transfer: TransferInfo = item.get("transferinfo")
        item_file_list = item_transfer.file_list_new

        for file_path in item_file_list:
            if os.path.splitext(file_path)[-1].lower() in settings.RMT_MEDIAEXT:
                self.add_task(file_path, TaskSource.EVENT)

    def _run_at_once(self, path_list: List[str], force_translate: bool = False):
        # 依次处理每个目录
        for path in path_list:
            if not os.path.exists(path) or not os.path.isabs(path):
                logger.warn(f"目录/文件无效，不进行处理:{path}")
                continue
            if os.path.isdir(path):
                for video_file in self.__get_library_files(path):
                    self.add_task(video_file, TaskSource.MANUAL, force_translate=force_translate)
            elif os.path.splitext(path)[-1].lower() in settings.RMT_MEDIAEXT:
                self.add_task(path, TaskSource.MANUAL, force_translate=force_translate)

    def __check_asr(self):
        if not self._faster_whisper_model_path or not self._faster_whisper_model:
            logger.warn(f"faster-whisper配置信息不完整，不进行处理")
            return False
        if not os.path.exists(self._faster_whisper_model_path):
            logger.info(f"创建faster-whisper模型目录：{self._faster_whisper_model_path}")
            os.mkdir(self._faster_whisper_model_path)
        try:
            from faster_whisper import WhisperModel, download_model
        except ImportError:
            logger.warn(f"faster-whisper 未安装，不进行处理")
            return False
        return True

    def __process_autosub(self, video_file, force_translate: bool = False) -> TaskStatus:
        if not video_file:
            return TaskStatus.FAILED
        # 如果文件大小小于指定大小， 则不处理
        if os.path.getsize(video_file) < self._file_size * 1024 * 1024:
            return TaskStatus.IGNORED

        start_time = time.time()
        file_path, file_ext = os.path.splitext(video_file)
        file_name = os.path.basename(video_file)

        try:
            # 字幕文件命名标签：强制翻译模式用"强制"，正常模式用"机翻"
            sub_tag = "强制" if force_translate else "机翻"
            logger.info(f"开始处理文件：{video_file} ..." + ("（强制翻译模式）" if force_translate else ""))
            # 判断目的字幕（和内嵌）是否已存在（强制翻译模式下跳过此检查）
            if not force_translate:
                if self.__target_subtitle_exists(video_file):
                    logger.warn(f"字幕文件已经存在，不进行处理")
                    return TaskStatus.IGNORED
            else:
                logger.info(f"强制翻译模式：跳过已有字幕检查，直接处理视频")
            # 生成字幕
            ret, lang, gen_sub_path = self.__generate_subtitle(video_file, file_path, self._enable_asr, sub_tag)
            if not ret:
                message = f" 媒体: {file_name}\n 生成字幕失败，跳过后续处理"
                if self._send_notify:
                    self.post_message(mtype=NotificationType.Plugin, title="【机器字幕生成】", text=message)
                return TaskStatus.FAILED

            if self._translate_zh:
                # 翻译字幕（先拿到双语条目数组，然后按三种模式保存）
                logger.info(f"开始翻译字幕为中文 ...")
                zh_sub_base = f"{file_path}.zh.{sub_tag}"
                # 统一翻译一次，得到双语字幕条目
                processed = self.__translate_zh_subtitle(lang, gen_sub_path)

                # 判断 gen_sub_path 是否为机器生成的（包含 机翻/强制 标签）
                # 只有机器生成的原字幕才会在双语模式下被删除；用户原外挂字幕不动
                gen_sub_str = str(gen_sub_path)
                is_machine_source = (f".{sub_tag}.srt" in gen_sub_str)

                mode = self._subtitle_mode
                output_files = []
                if mode == 'single':
                    # 单字幕：原声字幕（保留已存在的 gen_sub_path）+ 纯中文字幕（新文件）
                    zh_only_subs = self.__bilingual_to_chinese_only(processed)
                    zh_only_path = f"{zh_sub_base}.srt"
                    self.__save_srt(zh_only_path, zh_only_subs)
                    output_files.append(os.path.basename(zh_only_path))
                    if os.path.exists(gen_sub_str):
                        output_files.append(os.path.basename(gen_sub_str))
                elif mode == 'bilingual':
                    # 双语字幕：仅保留双语合并字幕；机器生成的原音字幕删除
                    bilingual_path = f"{zh_sub_base}.srt"
                    self.__save_srt(bilingual_path, processed)
                    output_files.append(os.path.basename(bilingual_path))
                    if is_machine_source and os.path.exists(gen_sub_str):
                        try:
                            os.remove(gen_sub_str)
                            logger.info(f"双语模式：已删除机器生成的原音字幕 {os.path.basename(gen_sub_str)}")
                        except Exception as e:
                            logger.debug(f"删除原音字幕失败 {gen_sub_str}: {e}")
                else:  # both
                    # 均保存：原声 + 纯中文 + 双语 三个文件全部保留
                    bilingual_path = f"{zh_sub_base}.双语.srt"
                    self.__save_srt(bilingual_path, processed)
                    output_files.append(os.path.basename(bilingual_path))

                    zh_only_subs = self.__bilingual_to_chinese_only(processed)
                    zh_only_path = f"{zh_sub_base}.srt"
                    self.__save_srt(zh_only_path, zh_only_subs)
                    output_files.append(os.path.basename(zh_only_path))

                    if os.path.exists(gen_sub_str):
                        output_files.append(os.path.basename(gen_sub_str))

                logger.info(f"翻译字幕完成（{mode}），输出文件：{', '.join(output_files)}")

            end_time = time.time()
            message = f" 媒体: {file_name}\n 处理完成\n 字幕原始语言: {lang}\n "
            if self._translate_zh:
                message += f"字幕翻译语言: zh, 模式: {self._subtitle_mode}\n "
            message += f"耗时：{round(end_time - start_time, 2)}秒"
            logger.info(f"机器字幕生成 处理完成：{message}")
            if self._send_notify:
                self.post_message(mtype=NotificationType.Plugin, title="【机器字幕生成】", text=message)
            return TaskStatus.COMPLETED
        except UserInterruptException:
            logger.info(f"用户中断当前任务：{video_file}")
            return TaskStatus.FAILED
        except Exception as e:
            logger.error(f"机器字幕生成 处理异常：{e}")
            end_time = time.time()
            message = f" 媒体: {file_name}\n 处理失败\n 耗时：{round(end_time - start_time, 2)}秒"
            if self._send_notify:
                self.post_message(mtype=NotificationType.Plugin, title="【机器字幕生成】", text=message)
            # 打印调用栈
            logger.error(traceback.format_exc())
            return TaskStatus.FAILED

    def __do_speech_recognition(self, audio_lang, audio_file):
        """
        语音识别, 生成字幕
        :param audio_lang:
        :param audio_file:
        :return:
        """
        lang = audio_lang
        try:
            from faster_whisper import WhisperModel, download_model
            # 设置缓存目录, 防止缓存同目录出现 cross-device 错误
            cache_dir = os.path.join(self._faster_whisper_model_path, "cache")
            if not os.path.exists(cache_dir):
                os.mkdir(cache_dir)
            os.environ["HF_HUB_CACHE"] = cache_dir
            if self._huggingface_proxy:
                os.environ["HTTP_PROXY"] = settings.PROXY['http']
                os.environ["HTTPS_PROXY"] = settings.PROXY['https']
            model = WhisperModel(
                download_model(self._faster_whisper_model, local_files_only=False, cache_dir=cache_dir),
                device="cpu", compute_type="int8", cpu_threads=psutil.cpu_count(logical=False))

            try:
                segments, info = model.transcribe(audio_file,
                                                  language=lang if lang != 'auto' else None,
                                                  word_timestamps=True,
                                                  vad_filter=True,
                                                  temperature=0,
                                                  beam_size=5)
                logger.info("Detected language '%s' with probability %f" % (info.language, info.language_probability))

                if lang == 'auto':
                    lang = info.language
            except ValueError as e:
                if "max() iterable argument is empty" in str(e):
                    logger.info("音频文件中未检测到任何语言内容，生成空字幕文件以避免重复处理")
                    # 生成空的字幕文件，避免重复识别
                    self.__save_srt(f"{audio_file}.srt", [])
                    # 如果原本是auto检测，设置一个默认语言
                    lang = 'und' if lang == 'auto' else lang
                    return True, lang
                else:
                    raise e

            subs = []
            if lang in ['en', 'eng']:
                # 英文先生成单词级别字幕，再合并
                idx = 0
                for segment in segments:
                    if self._event.is_set():
                        logger.info(f"whisper音轨转录服务停止")
                        raise UserInterruptException(f"用户中断当前任务")
                    for word in segment.words:
                        idx += 1
                        subs.append(srt.Subtitle(index=idx,
                                                 start=timedelta(seconds=word.start),
                                                 end=timedelta(seconds=word.end),
                                                 content=word.word))
                subs = self.__merge_srt(subs)
            else:
                for i, segment in enumerate(segments):
                    if self._event.is_set():
                        logger.info(f"whisper音轨转录服务停止")
                        raise UserInterruptException(f"用户中断当前任务")
                    subs.append(srt.Subtitle(index=i,
                                             start=timedelta(seconds=segment.start),
                                             end=timedelta(seconds=segment.end),
                                             content=segment.text))
            self.__save_srt(f"{audio_file}.srt", subs)
            logger.info(f"音轨转字幕完成")
            return True, lang
        except ImportError:
            logger.warn(f"faster-whisper 未安装，不进行处理")
            return False, None
        except Exception as e:
            traceback.print_exc()
            logger.error(f"faster-whisper 处理异常：{e}")
            return False, None

    def __generate_subtitle(self, video_file, subtitle_file, enable_asr=True, sub_tag="机翻"):
        """
        生成字幕
        :param video_file: 视频文件
        :param subtitle_file: 字幕文件, 不包含后缀
        :param sub_tag: 字幕命名标签（机翻/强制），用于机器生成的字幕文件名
        :return: 生成成功返回True，字幕语言,字幕路径，否则返回False, None, None
        """
        # 获取文件元数据
        video_meta = Ffmpeg().get_video_metadata(video_file)
        if not video_meta:
            logger.error(f"获取视频文件元数据失败，跳过后续处理")
            return False, None, None
        # 获取字幕语言偏好
        if self._translate_preference == "english_only":
            prefer_subtitle_langs = ['en', 'eng']
            strict = True
        elif self._translate_preference == "english_first":
            prefer_subtitle_langs = ['en', 'eng']
            strict = False
        else:  # self.translate_preference == "origin_first"
            prefer_subtitle_langs = None
            strict = False

        # 从视频文件音轨获取语言信息
        ret, audio_index, audio_lang = self.__get_video_prefer_audio(video_meta, prefer_lang=prefer_subtitle_langs)
        if not ret:
            logger.info(f"字幕源偏好：{self._translate_preference} 获取音轨元数据失败")
            return False, None, None

        # 如果开启了自动语言检测，直接设置为auto，跳过metadata的语言信息
        if self._auto_detect_language:
            logger.info("已开启自动语言检测，将使用whisper模型自动识别语言")
            audio_lang = 'auto'
        elif not iso639.find(audio_lang) or not iso639.to_iso639_1(audio_lang):
            logger.info(f"字幕源偏好：{self._translate_preference} 未从音轨元数据中获取到语言信息")
            audio_lang = 'auto'

        # 当字幕源偏好为origin_first时，优先使用音轨语言
        if self._translate_preference == "origin_first":
            prefer_subtitle_langs = ['en', 'eng'] if audio_lang == 'auto' else [audio_lang,
                                                                                iso639.to_iso639_1(audio_lang)]
        # 获取外挂字幕
        logger.info(f"使用 {prefer_subtitle_langs} 匹配已有外挂字幕文件 ...")
        external_sub_exist, external_sub_lang, exist_sub_name = self.__external_subtitle_exists(video_file,
                                                                                                prefer_subtitle_langs,
                                                                                                only_srt=True,
                                                                                                strict=strict)
        # 获取内嵌字幕
        logger.info(f"使用 {prefer_subtitle_langs} 匹配内嵌字幕文件 ...")
        inner_sub_exist, subtitle_index, inner_sub_lang, = self.__get_video_prefer_subtitle(video_meta,
                                                                                            prefer_subtitle_langs,
                                                                                            strict=strict)

        # 优先返回符合语言要求的外部字幕
        def get_sub_path():
            video_dir, _ = os.path.split(video_file)
            return os.path.join(video_dir, exist_sub_name)

        extract_subtitle = False
        if self._translate_preference == "english_only":
            if external_sub_exist:
                logger.info(f"字幕源偏好：{self._translate_preference} 外挂字幕存在，字幕语言 {external_sub_lang}")
                return True, iso639.to_iso639_1(external_sub_lang), get_sub_path()
            elif inner_sub_exist:
                logger.info(f"字幕源偏好：{self._translate_preference} 内嵌字幕存在，字幕语言 {inner_sub_lang}")
                extract_subtitle = True
            else:
                logger.info(f"字幕源偏好：{self._translate_preference} 未匹配到外挂或内嵌字幕,需要使用asr提取")
        else:  # english_first/origin_first
            if external_sub_exist and external_sub_lang in prefer_subtitle_langs:
                logger.info(f"字幕源偏好：{self._translate_preference} 外挂字幕存在，字幕语言 {external_sub_lang}")
                return True, iso639.to_iso639_1(external_sub_lang), get_sub_path()
            elif inner_sub_exist and inner_sub_lang in prefer_subtitle_langs:
                logger.info(f"字幕源偏好：{self._translate_preference} 内嵌字幕存在，字幕语言 {inner_sub_lang}")
                extract_subtitle = True
            elif external_sub_exist:
                logger.info(f"字幕源偏好：{self._translate_preference} 外挂字幕存在，字幕语言 {external_sub_lang}")
                return True, iso639.to_iso639_1(external_sub_lang), get_sub_path()
            elif inner_sub_exist:
                logger.info(f"字幕源偏好：{self._translate_preference} 内嵌字幕存在，字幕语言 {inner_sub_lang}")
                extract_subtitle = True
            else:
                logger.info(f"字幕源偏好：{self._translate_preference} 未匹配到外挂或内嵌字幕,需要使用asr提取")
        # 提取内嵌字幕
        if extract_subtitle:
            inner_sub_lang = iso639.to_iso639_1(inner_sub_lang) \
                if (inner_sub_lang and iso639.find(inner_sub_lang) and iso639.to_iso639_1(inner_sub_lang)) else 'und'
            extracted_sub_path = f"{subtitle_file}.{inner_sub_lang}.{sub_tag}.srt"
            Ffmpeg().extract_subtitle_from_video(video_file, extracted_sub_path, subtitle_index)
            logger.info(f"提取字幕完成：{extracted_sub_path}")
            return True, inner_sub_lang, extracted_sub_path
        # 使用asr音轨识别字幕
        if audio_lang != 'auto':
            audio_lang = iso639.to_iso639_1(audio_lang)

        if not enable_asr:
            logger.info(f"未开启语音识别，且无已有字幕文件，跳过后续处理")
            return False, None, None

        # 清理异常退出的临时文件
        tempdir = tempfile.gettempdir()
        try:
            for file in os.listdir(tempdir):
                if file.startswith('machinesub-'):
                    try:
                        os.remove(os.path.join(tempdir, file))
                    except Exception as e:
                        logger.debug(f"清理临时文件失败 {file}: {e}")
        except Exception as e:
            logger.debug(f"访问临时目录失败: {e}")

        with tempfile.NamedTemporaryFile(prefix='machinesub-', suffix='.wav', delete=True) as audio_file:
            # 提取音频
            logger.info(f"正在提取音频：{audio_file.name} ...")
            Ffmpeg().extract_wav_from_video(video_file, audio_file.name, audio_index)
            logger.info(f"提取音频完成：{audio_file.name}")

            # 生成字幕
            logger.info(f"开始生成字幕, 语言 {audio_lang} ...")
            # 临时 .srt 文件路径（__do_speech_recognition 内部会生成）
            temp_srt_path = f"{audio_file.name}.srt"
            try:
                ret, lang = self.__do_speech_recognition(audio_lang, audio_file.name)
                if ret:
                    logger.info(f"生成字幕成功，原始语言：{lang}")
                    # 复制字幕文件（机器生成的字幕加上标签：机翻/强制）
                    final_sub_path = f"{subtitle_file}.{lang}.{sub_tag}.srt"
                    SystemUtils.copy(Path(temp_srt_path), Path(final_sub_path))
                    logger.info(f"复制字幕文件：{final_sub_path}")
                    return ret, lang, Path(final_sub_path)
                else:
                    logger.error("生成字幕失败")
                    return False, None, None
            finally:
                # 确保临时 .srt 文件被清理（无论成功/失败/异常）
                # .wav 文件由 tempfile 的 delete=True 自动清理
                if os.path.exists(temp_srt_path):
                    try:
                        os.remove(temp_srt_path)
                    except Exception as e:
                        logger.debug(f"清理临时字幕文件失败 {temp_srt_path}: {e}")

    @staticmethod
    def __get_library_files(in_path, exclude_path=None):
        """
        获取目录媒体文件列表
        """
        if not os.path.isdir(in_path):
            yield in_path
            return

        for root, dirs, files in os.walk(in_path):
            if exclude_path and any(os.path.abspath(root).startswith(os.path.abspath(path))
                                    for path in exclude_path.split(",")):
                continue

            for file in files:
                cur_path = os.path.join(root, file)
                # 检查后缀
                if os.path.splitext(file)[-1].lower() in settings.RMT_MEDIAEXT:
                    yield cur_path

    @staticmethod
    def __load_srt(file_path):
        """
        加载字幕文件
        :param file_path: 字幕文件路径
        :return:
        """
        with open(file_path, 'r', encoding="utf8") as f:
            srt_text = f.read()
        return list(srt.parse(srt_text))

    @staticmethod
    def __save_srt(file_path, srt_data):
        """
        保存字幕文件
        :param file_path: 字幕文件路径
        :param srt_data: 字幕数据
        :return:
        """
        with open(file_path, 'w', encoding="utf8") as f:
            f.write(srt.compose(srt_data))

    @staticmethod
    def __bilingual_to_chinese_only(bilingual_subs: list) -> list:
        """
        将双语字幕（content = "中文\n原文"）转成仅中文的字幕。
        前提：翻译结果内部的换行已被替换为空格，保证第一个 \n 之前即为完整译文。
        """
        result = []
        for sub in bilingual_subs:
            # 用浅拷贝避免影响原数组
            content = sub.content or ""
            first_nl = content.find('\n')
            if first_nl >= 0:
                zh = content[:first_nl].strip()
            else:
                zh = content.strip()
            result.append(srt.Subtitle(index=sub.index, start=sub.start, end=sub.end, content=zh))
        return result

    def __merge_srt(self, subtitle_data):
        """
        合并整句字幕
        :param subtitle_data:
        :return:
        """
        subtitle_data = copy.deepcopy(subtitle_data)
        # 合并字幕
        merged_subtitle = []
        sentence_end = True
        end_tokens = ['.', '!', '?', '。', '！', '？', '。"', '！"', '？"', '."', '!"', '?"']
        for index, item in enumerate(subtitle_data):
            # 当前字幕先将多行合并为一行，再去除首尾空格
            content = item.content.replace('\n', ' ').strip()
            # 去除html标签
            parse = etree.HTML(content)
            if parse is not None:
                content = parse.xpath('string(.)')
            if content == '':
                continue
            item.content = content

            # 背景音等字幕，跳过
            if self.__is_noisy_subtitle(content):
                merged_subtitle.append(item)
                sentence_end = True
                continue

            if not merged_subtitle or sentence_end:
                merged_subtitle.append(item)
            elif not sentence_end:
                merged_subtitle[-1].content = f"{merged_subtitle[-1].content} {content}"
                merged_subtitle[-1].end = item.end

            # 如果当前字幕内容以标志符结尾，则设置语句已经终结
            if content.endswith(tuple(end_tokens)):
                sentence_end = True
            # 如果上句字幕超过一定长度，则设置语句已经终结
            elif len(merged_subtitle[-1].content) > 80:
                sentence_end = True
            else:
                sentence_end = False

        return merged_subtitle

    @staticmethod
    def __get_video_prefer_audio(video_meta, prefer_lang=None):
        """
        获取视频的首选音轨，如果有多音轨， 优先指定语言音轨，否则获取默认音轨
        :param video_meta
        :return:
        """
        if type(prefer_lang) == str and prefer_lang:
            prefer_lang = [prefer_lang]

        # 获取首选音轨
        audio_lang = None
        audio_index = None
        audio_stream = filter(lambda x: x.get('codec_type') == 'audio', video_meta.get('streams', []))
        for index, stream in enumerate(audio_stream):
            if not audio_index:
                audio_index = index
                audio_lang = stream.get('tags', {}).get('language', 'und')
            # 获取默认音轨
            if stream.get('disposition', {}).get('default'):
                audio_index = index
                audio_lang = stream.get('tags', {}).get('language', 'und')
            # 获取指定语言音轨
            if prefer_lang and stream.get('tags', {}).get('language') in prefer_lang:
                audio_index = index
                audio_lang = stream.get('tags', {}).get('language', 'und')
                break

        # 如果没有音轨， 则不处理
        if audio_index is None:
            logger.warn(f"没有音轨，不进行处理")
            return False, None, None

        logger.info(f"选中音轨信息：{audio_index}, {audio_lang}")
        return True, audio_index, audio_lang

    @staticmethod
    def __get_video_prefer_subtitle(video_meta, prefer_lang=None, strict=False, only_srt=True):
        """
        获取视频的首选字幕。优先级：1.字幕为偏好语言 2.默认字幕 3.第一个字幕
        :param video_meta: 视频元数据
        :param prefer_lang: 字幕偏好语言
        :param strict: 是否严格模式。如果指定了偏好语言，严格模式下必须返回偏好语言的字幕。
        :return: (是否命中字幕，字幕index，字幕语言)
        """
        image_based_subtitle_codecs = (
            'dvd_subtitle',
            'dvb_subtitle',
            'hdmv_pgs_subtitle',
        )

        if prefer_lang is str and prefer_lang:
            prefer_lang = [prefer_lang]

        # 获取首选字幕
        subtitle_lang = None
        subtitle_index = None
        subtitle_score = 0
        subtitle_stream = filter(lambda x: x.get('codec_type') == 'subtitle', video_meta.get('streams', []))
        for index, stream in enumerate(subtitle_stream):
            # 如果是强制字幕，则跳过
            if stream.get('disposition', {}).get('forced'):
                continue
            # image-based 字幕，跳过
            if only_srt and (
                    'width' in stream
                    or stream.get('codec_name') in image_based_subtitle_codecs
            ):
                continue
            cur_is_default = stream.get('disposition', {}).get('default')
            cur_lang = stream.get('tags', {}).get('language')
            # 计算当前字幕得分：1.字幕为偏好语言*4 2.默认字幕*2 3.第一个字幕*1
            cur_score = 0
            if prefer_lang and cur_lang in prefer_lang:
                cur_score += 4
            if cur_is_default:
                cur_score += 2
            if subtitle_index is None:
                cur_score += 1
                # 第一个字幕初始化为默认字幕
                subtitle_lang, subtitle_index, subtitle_score = cur_lang, index, cur_score
            if cur_score > subtitle_score:
                subtitle_lang, subtitle_index, subtitle_score = cur_lang, index, cur_score

        # 未找到字幕
        if subtitle_index is None:
            logger.debug(f"没有内嵌字幕")
            return False, None, None
        if strict and prefer_lang and subtitle_lang not in prefer_lang:
            logger.warn(f"严格模式,没有偏好语言的字幕")
            return False, None, None
        logger.debug(f"命中内嵌字幕信息：{subtitle_index}, {subtitle_lang}, score:{subtitle_score}")
        return True, subtitle_index, subtitle_lang

    @staticmethod
    def __is_noisy_subtitle(content):
        """
        判断是否为背景音等字幕
        :param content:
        :return:
        """
        noisy_tokens = [('(', ')'), ('[', ']'), ('{', '}'), ('【', '】'), ('♪', '♪'), ('♫', '♫'), ('♪♪', '♪♪')]
        return any(content.startswith(t[0]) and content.endswith(t[1]) for t in noisy_tokens)

    def __get_context(self, all_subs: list, target_indices: List[int], is_batch: bool) -> str:
        """通用上下文获取方法"""
        min_idx = max(0, min(target_indices) - self._context_window)
        max_idx = min(len(all_subs) - 1, max(target_indices) + self._context_window) if is_batch else min(
            target_indices)

        context = []
        for idx in range(min_idx, max_idx + 1):
            status = "[待译]" if idx in target_indices else ""
            content = all_subs[idx].content.replace('\n', ' ').strip()
            context.append(f"{status}{content}")

        return "\n".join(context)

    def __process_items(self, all_subs: list, items: list, bilingual: bool = True) -> list:
        """统一处理入口（支持批量和单条）"""
        if self._enable_batch and len(items) > 1:
            return self.__process_batch(all_subs, items, bilingual)
        return [self.__process_single(all_subs, item, bilingual) for item in items]

    def __translate_to_zh(self, text: str, context: str = None) -> str:
        if self._event.is_set():
            raise UserInterruptException("用户中断当前任务")
        if self._translate_service == 'volcengine':
            return self._volcengine_translate.translate_to_zh(text, context, max_retries=self._max_retries)
        else:
            return self._baidu_translate.translate_to_zh(text, context, max_retries=self._max_retries)

    def __process_batch(self, all_subs: list, batch: list, bilingual: bool = True) -> list:
        """批量处理逻辑"""
        indices = [all_subs.index(item) for item in batch]
        context = self.__get_context(all_subs, indices, is_batch=True) if self._context_window > 0 else None
        # 关键：去除每条字幕内部的换行符，确保 batch_text 的行数 == batch 大小
        # 否则多行字幕会让百度翻译返回的行数对不上，触发"行数不匹配"
        batch_text = '\n'.join([item.content.replace('\n', ' ').strip() for item in batch])

        try:
            ret, result = self.__translate_to_zh(batch_text, context)
            if not ret:
                raise Exception(result)

            translated = [line.strip() for line in result.split('\n') if line.strip()]
            if len(translated) != len(batch):
                raise Exception(f"批次行数不匹配 {len(translated)}/{len(batch)}")

            for item, trans in zip(batch, translated):
                # 双语模式：中文 + 原文；单字幕模式：仅中文
                item.content = f"{trans}\n{item.content}" if bilingual else trans
            self._stats['batch_success'] += len(batch)
            return batch
        except Exception as e:
            logger.warning(f"批次翻译失败（{str(e)}），降级到单行匹配...")
            self._stats['batch_fail'] += 1
            return [self.__process_single(all_subs, item, bilingual) for item in batch]

    def __process_single(self, all_subs: List[srt.Subtitle], item: srt.Subtitle,
                         bilingual: bool = True) -> srt.Subtitle:
        """单条处理逻辑"""
        idx = all_subs.index(item)
        context = self.__get_context(all_subs, [idx], is_batch=False) if self._context_window > 0 else None
        success, trans = self.__translate_to_zh(item.content, context)

        if success:
            # 双语模式：中文 + 原文；单字幕模式：仅中文
            item.content = f"{trans}\n{item.content}" if bilingual else trans
            self._stats['line_fallback'] += 1
            return item
        else:
            item.content = f"[翻译失败]\n{item.content}" if bilingual else "[翻译失败]"
            return item

    def __translate_zh_subtitle(self, source_lang: str, source_subtitle: str,
                                dest_bilingual: str = None) -> list:
        """
        翻译字幕为中文，返回双语字幕条目数组（每条 content = "中文\n原文"）。
        不负责保存文件，保存逻辑由调用方按三种模式处理。
        :param source_lang: 源字幕语言
        :param source_subtitle: 源字幕文件路径
        :param dest_bilingual: 若提供则保存一份双语字幕到此路径（方便直接取用）
        :return: processed 双语字幕条目 list（可再加工）
        """
        self._stats = {'total': 0, 'batch_success': 0, 'batch_fail': 0, 'line_fallback': 0}
        subs = self.__load_srt(source_subtitle)
        if source_lang in ["en", "eng"] and self._enable_merge:
            valid_subs = self.__merge_srt(subs)
            logger.info(f"英文字幕合并：合并前字幕数: {len(subs)},合并后字幕数: {len(valid_subs)}")
        else:
            valid_subs = subs

        if not valid_subs:
            logger.warning("字幕文件为空或没有有效的字幕条目，跳过翻译")
            if dest_bilingual:
                self.__save_srt(dest_bilingual, [])
            return []

        # 始终按 bilingual=True 翻译，保证每个条目里中文+原文都在，后续可灵活切分
        _bilingual = True

        self._stats['total'] = len(valid_subs)
        processed = []
        current_batch = []

        for item in valid_subs:
            current_batch.append(item)

            if len(current_batch) >= self._batch_size:
                processed += self.__process_items(valid_subs, current_batch, _bilingual)
                current_batch = []
                logger.info(f"进度: {len(processed)}/{len(valid_subs)}")

        if current_batch:
            processed += self.__process_items(valid_subs, current_batch, _bilingual)

        # 若指定了双语输出路径，则直接保存（调用方后续再决定是否留此文件）
        if dest_bilingual:
            self.__save_srt(dest_bilingual, processed)

        success_rate = (self._stats['batch_success'] / self._stats['total'] * 100) if self._stats['total'] > 0 else 0.0

        logger.info(f"""
    翻译完成！
    总处理条目: {self._stats['total']}
    批次成功: {self._stats['batch_success']} ({success_rate:.1f}%)
    批次失败: {self._stats['batch_fail']}
    行补偿翻译: {self._stats['line_fallback']}
            """)

        return processed

    @staticmethod
    def __external_subtitle_exists(video_file, prefer_langs=None, only_srt=False, strict=True):
        """
        外部字幕文件是否存在,支持多种格式及扩展需求。
        :param video_file: 视频文件路径
        :param prefer_langs: 偏好语言列表，支持单个语言字符串或列表
        :param only_srt: 是否只匹配srt格式的字幕
        :param strict: 是否严格匹配偏好语言.当不存在偏好语言字幕但存在其他语言字幕时,是否返回其他字幕
        :return: 元组 (是否存在, 检测到的语言, 文件名)
        """
        video_dir, video_name = os.path.split(video_file)
        video_name, video_ext = os.path.splitext(video_name)

        if prefer_langs and type(prefer_langs) == str:
            prefer_langs = [prefer_langs]

        metadata_flags = ["default", "forced", "foreign", "sdh", "cc", "hi", "机翻", "强制", "双语"]
        if only_srt:
            subtitle_extensions = [".srt"]
        else:
            subtitle_extensions = [".srt", ".sub", ".ass", ".ssa", ".vtt"]

        def parse_props(props):
            """
            解析字幕属性信息，提取语言和元数据标记。
            :param props: 属性字符串
            :return: (语言, 元数据列表)
            """
            parts = props.split(".")
            if len(parts) < 1:
                return None, []

            cur_subtitle_lang = None
            cur_metadata = []
            # 倒序遍历文件名中的标记
            for i in range(len(parts) - 1, -1, -1):
                part = parts[i]
                if part in metadata_flags:
                    cur_metadata.append(part)
                elif cur_subtitle_lang is None:
                    try:
                        iso639.to_iso639_1(part)
                    except iso639.NonExistentLanguageError:
                        continue
                    else:
                        cur_subtitle_lang = iso639.to_iso639_1(part)  # 记录最后一个语言标记

            return cur_subtitle_lang, cur_metadata

        # 备选的字幕语言.当strict=False时生效, 用于在未找到偏好语言时返回其他语言
        second_lang = None
        second_file = None
        # 检查字幕文件
        for file in os.listdir(video_dir):
            if not file.startswith(video_name):
                continue

            # 检查扩展名是否在支持范围内
            _, ext = os.path.splitext(file)
            if ext.lower() not in subtitle_extensions:
                continue

            # 提取文件名中的语言和元数据信息
            props_str = file[len(video_name) + 1: -len(ext)] if file.startswith(video_name + ".") else ""
            subtitle_lang, metadata = parse_props(props_str)

            # 如果没有语言标记，跳过
            if not subtitle_lang:
                continue

            # 如果指定了偏好语言
            if prefer_langs:
                if subtitle_lang in prefer_langs:
                    return True, subtitle_lang, file
                else:
                    second_lang = subtitle_lang
                    second_file = file
            else:
                # 未指定偏好语言，找到的第一个字幕即返回
                return True, subtitle_lang, file
        if not strict and second_lang:
            return True, second_lang, second_file
        return False, None, None

    def __target_subtitle_exists(self, video_file):
        """
        目标字幕文件是否存在
        :param video_file:
        :return:
        """
        if self._translate_zh:
            prefer_langs = ['zh', 'chi', 'zh-CN', 'chs', 'zhs', 'zh-Hans', 'zhong', 'simp', 'cn']
            strict = True
        else:
            if self._translate_preference == "english_first":
                prefer_langs = ['en', 'eng']
                strict = False
            elif self._translate_preference == "english_only":
                prefer_langs = ['en', 'eng']
                strict = True
            else:
                prefer_langs = None
                strict = False

        exist, lang, _ = self.__external_subtitle_exists(video_file, prefer_langs, strict=strict)
        if exist:
            return True

        video_meta = Ffmpeg().get_video_metadata(video_file)
        if not video_meta:
            return False
        ret, subtitle_index, subtitle_lang = self.__get_video_prefer_subtitle(video_meta, prefer_lang=prefer_langs,
                                                                              only_srt=False)
        if ret and subtitle_lang in prefer_langs:
            return True

        return False

    def get_form(self) -> Tuple[List[dict], Dict[str, Any]]:
        """
        拼装插件配置页面，需要返回两块数据：1、页面配置；2、数据结构
        """
        return [
            {
                'component': 'VForm',
                'content': [
                    {
                        'component': 'VRow',
                        'content': [
                            {
                                'component': 'VCol',
                                'props': {'cols': 12, 'md': 4},
                                'content': [
                                    {
                                        'component': 'VSwitch',
                                        'props': {
                                            'model': 'enabled',
                                            'label': '启用插件',
                                            'color': 'primary'
                                        }
                                    }
                                ]
                            },
                            {
                                'component': 'VCol',
                                'props': {'cols': 12, 'md': 4},
                                'content': [
                                    {
                                        'component': 'VSwitch',
                                        'props': {
                                            'model': 'clear_history',
                                            'label': '清理历史记录',
                                        }
                                    }
                                ]
                            },
                            {
                                'component': 'VCol',
                                'props': {'cols': 12, 'md': 4},
                                'content': [
                                    {
                                        'component': 'VSwitch',
                                        'props': {
                                            'model': 'send_notify',
                                            'label': '发送通知'
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        'component': 'VRow',
                        'content': [
                            {
                                'component': 'VCol',
                                'props': {'cols': 12, 'md': 4},
                                'content': [
                                    {
                                        'component': 'VSwitch',
                                        'props': {
                                            'model': 'listen_transfer_event',
                                            'label': '媒体入库自动执行',
                                            'hint': '监听媒体入库事件，自动执行字幕生成'
                                        }
                                    }
                                ]
                            },
                            {
                                'component': 'VCol',
                                'props': {'cols': 12, 'md': 4},
                                'content': [
                                    {
                                        'component': 'VSwitch',
                                        'props': {
                                            'model': 'run_now',
                                            'label': '手动执行一次',
                                            'color': 'secondary'
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        'component': 'VRow',
                        'props': {'v-show': 'run_now'},
                        'content': [
                            {
                                'component': 'VCol',
                                'props': {'cols': 12, 'md': 12},
                                'content': [
                                    {
                                        'component': 'VTextarea',
                                        'props': {
                                            'model': 'path_list',
                                            'label': '媒体路径',
                                            'rows': 3,
                                            'placeholder': '绝对路径，每行一个。支持文件和文件夹'
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        'component': 'VRow',
                        'props': {'v-show': 'run_now'},
                        'content': [
                            {
                                'component': 'VCol',
                                'props': {'cols': 12, 'md': 12},
                                'content': [
                                    {
                                        'component': 'VSwitch',
                                        'props': {
                                            'model': 'force_translate',
                                            'label': '强制翻译',
                                            'color': 'error',
                                            'hint': '忽略已有字幕检查，直接对视频进行音频提取、字幕生成和翻译'
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        'component': 'VRow',
                        'content': [
                            {
                                'component': 'VCol',
                                'props': {'cols': 12, 'md': 4},
                                'content': [
                                    {
                                        'component': 'VTextField',
                                        'props': {
                                            'model': 'file_size',
                                            'label': '触发字幕生成的视频文件不小于(MB)',
                                            'placeholder': '默认10'
                                        }
                                    }
                                ]
                            },
                            {
                                'component': 'VCol',
                                'props': {'cols': 12, 'md': 4},
                                'content': [
                                    {
                                        'component': 'VSelect',
                                        'props': {
                                            'model': 'translate_preference',
                                            'label': '字幕源语言偏好',
                                            'hint': '小语种视频存在多语言字幕/音轨时，优先选择哪种语言用于翻译',
                                            'items': [
                                                {'title': '仅英文', 'value': 'english_only'},
                                                {'title': '英文优先', 'value': 'english_first'},
                                                {'title': '原音优先', 'value': 'origin_first'}
                                            ]
                                        }
                                    }
                                ]
                            },
                            {
                                'component': 'VCol',
                                'props': {'cols': 12, 'md': 4},
                                'content': [
                                    {
                                        'component': 'VSwitch',
                                        'props': {
                                            'model': 'translate_zh',
                                            'label': '翻译成中文',
                                            'hint': '使用机器翻译将字幕翻译成中文'
                                        }
                                    }
                                ]
                            },
                            {
                                'component': 'VCol',
                                'props': {
                                    'cols': 12,
                                    'md': 6,
                                    'v-show': 'translate_zh'
                                },
                                'content': [
                                    {
                                        'component': 'VSelect',
                                        'props': {
                                            'model': 'subtitle_mode',
                                            'label': '字幕输出模式',
                                            'hint': '单字幕=原声字幕+纯中文；双语=仅中文+原文合并；均保存=三个文件都保留',
                                            'items': [
                                                {'title': '单字幕（原声字幕 + 纯中文字幕，两个文件）', 'value': 'single'},
                                                {'title': '双语字幕（中文+原文合并，一个文件）', 'value': 'bilingual'},
                                                {'title': '均保存（原声 + 纯中文 + 双语，三个文件）', 'value': 'both'}
                                            ]
                                        }
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        'component': 'VCard',
                        'props': {'variant': 'outlined', 'class': 'mb-4'},
                        'content': [
                            {
                                'component': 'VCardTitle',
                                'text': 'ASR配置（语音识别）'
                            },
                            {
                                'component': 'VCardText',
                                'content': [
                                    {
                                        'component': 'VRow',
                                        'content': [
                                            {
                                                'component': 'VCol',
                                                'props': {'cols': 12, 'md': 4},
                                                'content': [
                                                    {
                                                        'component': 'VSwitch',
                                                        'props': {
                                                            'model': 'enable_asr',
                                                            'label': '允许从音轨生成字幕'
                                                        }
                                                    }
                                                ]
                                            },
                                            {
                                                'component': 'VCol',
                                                'props': {'cols': 12, 'md': 4},
                                                'content': [
                                                    {
                                                        'component': 'VSwitch',
                                                        'props': {
                                                            'model': 'auto_detect_language',
                                                            'label': '自动检测语言',
                                                            'hint': '使用whisper模型自动检测语言，而非依赖视频元数据'
                                                        }
                                                    }
                                                ]
                                            },
                                            {
                                                'component': 'VCol',
                                                'props': {'cols': 12, 'md': 4},
                                                'content': [
                                                    {
                                                        'component': 'VSelect',
                                                        'props': {
                                                            'model': 'faster_whisper_model',
                                                            'label': 'faster-whisper模型选择',
                                                            'items': ['tiny', 'base', 'small', 'medium',
                                                                      'large-v3',
                                                                      {'title': 'large-v3-turbo',
                                                                       'value': 'deepdml/faster-whisper-large-v3-turbo-ct2'},
                                                                      ]
                                                        }
                                                    }
                                                ]
                                            }
                                        ]
                                    },
                                    {
                                        'component': 'VRow',
                                        'content': [
                                            {
                                                'component': 'VCol',
                                                'props': {'cols': 12, 'md': 12},
                                                'content': [
                                                    {
                                                        'component': 'VSwitch',
                                                        'props': {
                                                            'model': 'proxy',
                                                            'hint': '需配置MP环境变量PROXY_HOST',
                                                            'label': '使用代理下载模型'
                                                        }
                                                    }
                                                ]
                                            }
                                        ]
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        'component': 'VCard',
                        'props': {'variant': 'outlined', 'class': 'mb-4'},
                        'content': [
                            {
                                'component': 'VCardTitle',
                                'text': '翻译服务设置'
                            },
                            {
                                'component': 'VCardText',
                                'content': [
                                    {
                                        'component': 'VRow',
                                        'content': [
                                            {
                                                'component': 'VCol',
                                                'props': {
                                                    'cols': 12,
                                                    'md': 6
                                                },
                                                'content': [
                                                    {
                                                        'component': 'VSelect',
                                                        'props': {
                                                            'model': 'translate_service',
                                                            'label': '翻译服务',
                                                            'items': [
                                                                {'title': '百度翻译', 'value': 'baidu'},
                                                                {'title': '火山引擎翻译', 'value': 'volcengine'}
                                                            ]
                                                        }
                                                    }
                                                ]
                                            }
                                        ]
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        'component': 'VCard',
                        'props': {'variant': 'outlined', 'class': 'mb-4'},
                        'content': [
                            {
                                'component': 'VCardTitle',
                                'text': '百度翻译接口设置'
                            },
                            {
                                'component': 'VCardText',
                                'content': [
                                    {
                                        'component': 'VRow',
                                        'content': [
                                            {
                                                'component': 'VCol',
                                                'props': {
                                                    'cols': 12,
                                                    'md': 6
                                                },
                                                'content': [
                                                    {
                                                        'component': 'VTextField',
                                                        'props': {
                                                            'model': 'baidu_appid',
                                                            'label': '百度翻译APP ID',
                                                            'placeholder': '请输入百度翻译APP ID',
                                                            'hint': '在百度翻译开放平台申请: https://fanyi-api.baidu.com'
                                                        }
                                                    }
                                                ]
                                            },
                                            {
                                                'component': 'VCol',
                                                'props': {
                                                    'cols': 12,
                                                    'md': 6
                                                },
                                                'content': [
                                                    {
                                                        'component': 'VTextField',
                                                        'props': {
                                                            'model': 'baidu_secret_key',
                                                            'label': '百度翻译密钥',
                                                            'placeholder': '请输入百度翻译密钥'
                                                        }
                                                    }
                                                ]
                                            }
                                        ]
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        'component': 'VCard',
                        'props': {'variant': 'outlined', 'class': 'mb-4'},
                        'content': [
                            {
                                'component': 'VCardTitle',
                                'text': '火山引擎翻译接口设置'
                            },
                            {
                                'component': 'VCardText',
                                'content': [
                                    {
                                        'component': 'VRow',
                                        'content': [
                                            {
                                                'component': 'VCol',
                                                'props': {
                                                    'cols': 12,
                                                    'md': 6
                                                },
                                                'content': [
                                                    {
                                                        'component': 'VTextField',
                                                        'props': {
                                                            'model': 'volc_access_key',
                                                            'label': '火山引擎Access Key',
                                                            'placeholder': '请输入Access Key',
                                                            'hint': '在火山引擎控制台申请: https://www.volcengine.com/docs/4640/65067'
                                                        }
                                                    }
                                                ]
                                            },
                                            {
                                                'component': 'VCol',
                                                'props': {
                                                    'cols': 12,
                                                    'md': 6
                                                },
                                                'content': [
                                                    {
                                                        'component': 'VTextField',
                                                        'props': {
                                                            'model': 'volc_secret_key',
                                                            'label': '火山引擎Secret Key',
                                                            'placeholder': '请输入Secret Key'
                                                        }
                                                    }
                                                ]
                                            }
                                        ]
                                    },
                                ]
                            }
                        ]
                    },
                    {
                        'component': 'VCard',
                        'props': {'variant': 'outlined', 'class': 'mb-4'},
                        'content': [
                            {
                                'component': 'VCardTitle',
                                'text': '翻译参数设置'
                            },
                            {
                                'component': 'VCardText',
                                'content': [
                                    {
                                        'component': 'VRow',
                                        'content': [
                                            {
                                                'component': 'VCol',
                                                'props': {'cols': 12, 'md': 4},
                                                'content': [
                                                    {
                                                        'component': 'VTextField',
                                                        'props': {
                                                            'model': 'context_window',
                                                            'label': '上下文窗口大小',
                                                            'placeholder': '0',
                                                            'hint': '翻译API不支持上下文，建议设为0'
                                                        }
                                                    }
                                                ]
                                            },
                                            {
                                                'component': 'VCol',
                                                'props': {'cols': 12, 'md': 4},
                                                'content': [
                                                    {
                                                        'component': 'VTextField',
                                                        'props': {
                                                            'model': 'max_retries',
                                                            'label': '翻译请求重试次数',
                                                            'placeholder': '3'
                                                        }
                                                    }
                                                ]
                                            },
                                            {
                                                'component': 'VCol',
                                                'props': {'cols': 12, 'md': 4},
                                                'content': [
                                                    {
                                                        'component': 'VSwitch',
                                                        'props': {
                                                            'model': 'enable_merge',
                                                            'label': '翻译英文时合并整句'
                                                        }
                                                    }
                                                ]
                                            }
                                        ]
                                    },
                                    {
                                        'component': 'VRow',
                                        'content': [
                                            {
                                                'component': 'VCol',
                                                'props': {'cols': 12, 'md': 4},
                                                'content': [
                                                    {
                                                        'component': 'VSwitch',
                                                        'props': {
                                                            'model': 'enable_batch',
                                                            'label': '启用批量翻译'
                                                        }
                                                    }
                                                ]
                                            },
                                            {
                                                'component': 'VCol',
                                                'props': {'cols': 12, 'md': 4},
                                                'content': [
                                                    {
                                                        'component': 'VTextField',
                                                        'props': {
                                                            'model': 'batch_size',
                                                            'label': '每批翻译行数',
                                                            'placeholder': '10'
                                                        }
                                                    }
                                                ]
                                            }
                                        ]
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        'component': 'VRow',
                        'content': [
                            {
                                'component': 'VCol',
                                'props': {'cols': 12},
                                'content': [
                                    {
                                        'component': 'VAlert',
                                        'props': {
                                            'type': 'info',
                                            'variant': 'tonal'
                                        },
                                        'content': [
                                            {
                                                'component': 'span',
                                                'text': '强制翻译说明：仅在「手动执行一次」打开时可用。启用后将忽略已有字幕检查，直接对视频进行音频提取、字幕生成和翻译，适用于需要重新生成或覆盖已有字幕的场景。'
                                            }
                                        ]
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        ], {
            "enabled": False,
            "clear_history": False,
            "send_notify": False,
            "listen_transfer_event": True,
            "run_now": False,
            "force_translate": False,
            "path_list": "",
            "file_size": "10",
            "translate_preference": "english_first",
            "translate_zh": False,
            "translate_service": "baidu",
            "enable_asr": True,
            "auto_detect_language": False,
            "faster_whisper_model": "base",
            "proxy": True,
            "baidu_appid": None,
            "baidu_secret_key": None,
            "volc_access_key": None,
            "volc_secret_key": None,
            "context_window": 0,
            "max_retries": 3,
            "enable_merge": False,
            "subtitle_mode": "bilingual",
            "enable_batch": True,
            "batch_size": 10,
        }

    def get_api(self) -> List[Dict[str, Any]]:
        return []

    def get_page(self) -> List[dict]:
        # 加载任务并按添加时间倒序排列
        tasks: Dict[str, TaskItem] = self.load_tasks()
        sorted_tasks = sorted(
            tasks.items(),
            key=lambda x: x[1].add_time,
            reverse=True
        )

        status_classes = {
            TaskStatus.PENDING: "text-info",
            TaskStatus.IN_PROGRESS: "text-warning",
            TaskStatus.COMPLETED: "text-success",
            TaskStatus.IGNORED: "text-muted",
            TaskStatus.FAILED: "text-error"
        }

        rows = []
        for task_id, task in sorted_tasks:
            source_label = {
                TaskSource.MANUAL: "手动添加",
                TaskSource.EVENT: "入库触发"
            }.get(task.source, task.source)

            status_text = {
                TaskStatus.PENDING: "等待中",
                TaskStatus.IN_PROGRESS: "处理中",
                TaskStatus.COMPLETED: "已完成",
                TaskStatus.IGNORED: "已忽略",
                TaskStatus.FAILED: "失败"
            }.get(task.status, task.status)

            status_class = status_classes.get(task.status, "")

            add_time_str = task.add_time.strftime("%Y-%m-%d %H:%M:%S")
            complete_time_str = (
                task.complete_time.strftime("%Y-%m-%d %H:%M:%S")
                if task.complete_time else "-"
            )

            # 强制翻译标记
            force_label = "（强制）" if task.force_translate else ""

            rows.append({
                "component": "tr",
                "props": {"class": "text-sm"},
                "content": [
                    {"component": "td", "text": add_time_str},
                    {"component": "td", "text": task.video_file},
                    {"component": "td", "text": f"{source_label}{force_label}"},
                    {"component": "td", "text": complete_time_str},
                    {
                        "component": "td",
                        "props": {"class": status_class},
                        "text": status_text
                    },
                ],
            })

        return [
            {
                "component": "VRow",
                "content": [
                    {
                        "component": "VCol",
                        "props": {"cols": 12},
                        "content": [
                            {
                                "component": "VTable",
                                "props": {"hover": True},
                                "content": [
                                    {
                                        "component": "thead",
                                        "content": [
                                            {
                                                "component": "th",
                                                "props": {"class": "text-start ps-4"},
                                                "text": "添加时间"
                                            },
                                            {
                                                "component": "th",
                                                "props": {"class": "text-start ps-4"},
                                                "text": "视频文件"
                                            },
                                            {
                                                "component": "th",
                                                "props": {"class": "text-start ps-4"},
                                                "text": "来源"
                                            },
                                            {
                                                "component": "th",
                                                "props": {"class": "text-start ps-4"},
                                                "text": "完成时间"
                                            },
                                            {
                                                "component": "th",
                                                "props": {"class": "text-start ps-4"},
                                                "text": "状态"
                                            },
                                        ]
                                    },
                                    {"component": "tbody", "content": rows}
                                ]
                            }
                        ]
                    }
                ]
            }
        ]

    @staticmethod
    def get_command() -> List[Dict[str, Any]]:
        return []

    def get_state(self) -> bool:
        """
        获取插件状态，如果插件正在运行， 则返回True
        """
        return self._running

    def stop_service(self):
        """
        退出插件
        """
        if self._running:
            self._event.set()
        if self._consumer_thread and self._consumer_thread.is_alive():
            logger.info("正在停止当前任务...")
            self._consumer_thread.join()

        if self._task_queue:
            while not self._task_queue.empty():
                self._task_queue.get_nowait()
                self._task_queue.task_done()
            logger.info("任务队列已清空")
        if self._tasks is not None:
            for task_id in list(self._tasks.keys()):
                task = self._tasks[task_id]
                if task.status == TaskStatus.PENDING or task.status == TaskStatus.IN_PROGRESS:
                    task.status = TaskStatus.FAILED
                    task.complete_time = datetime.now()
            self.save_tasks()  # 持久化更新后的任务列表
        self._running = False
        self._event.clear()
        logger.info(f"机器字幕生成与翻译服务已停止")
