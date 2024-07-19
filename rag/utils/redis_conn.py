# 导入json模块，用于数据的序列化和反序列化
import json

# 导入redis模块，用于操作Redis数据库
import redis
# 导入logging模块，用于日志记录
import logging
# 从rag模块导入settings，用于访问应用程序的配置设置
from rag import settings
# 从rag模块的utils子模块导入singleton，用于实现单例模式
from rag.utils import singleton


class Payload:
    def __init__(self, consumer, queue_name, group_name, msg_id, message):
        """
        初始化ConsumerRecord对象。

        参数:
        consumer - 消费者对象，负责消息的消费。
        queue_name - 队列名称，消息所在的队列。
        group_name - 消费者组名称，用于标识消息消费者群体。
        msg_id - 消息ID，唯一标识一条消息。
        message - 包含消息内容的字典对象。

        """
        self.__consumer = consumer
        self.__queue_name = queue_name
        self.__group_name = group_name
        self.__msg_id = msg_id
        # 将消息内容从JSON字符串解析为Python对象
        self.__message = json.loads(message['message'])

    def ack(self):
        """
        确认消息消费。

        尝试调用消费者对象的xack方法，确认消息已被处理。如果确认成功，返回True；
        如果出现异常，记录警告日志并返回False。

        Returns:
            bool: 确认成功返回True，否则返回False。
        """
        try:
            # 尝试确认消息消费
            self.__consumer.xack(self.__queue_name, self.__group_name, self.__msg_id)
            return True
        except Exception as e:
            # 记录异常信息
            logging.warning("[EXCEPTION]ack" + str(self.__queue_name) + "||" + str(e))
        return False

    def get_message(self):
        """
        获取私有变量__message的值。

        由于__message被设计为私有变量，因此这个方法提供了一个安全的方式来访问它的值，
        而不直接暴露变量本身，这有助于封装和数据隐藏。

        返回:
            str: __message变量的当前值。
        """
        return self.__message


@singleton
class RedisDB:
    def __init__(self):
        """
        初始化Redis连接客户端。

        本方法在类实例化时自动调用，用于设置Redis连接的相关属性并建立连接。
        """
        # 初始化Redis连接对象为None
        self.REDIS = None
        # 从settings配置文件中获取Redis配置信息
        self.config = settings.REDIS
        # 调用私有方法__open__建立Redis连接
        self.__open__()

    def __open__(self):
        """
        私有方法，用于初始化Redis连接。

        根据配置信息尝试建立与Redis服务器的连接。如果配置中未指定端口，则默认使用6379；
        如果未指定数据库编号，则默认使用1。支持通过配置文件设置Redis的密码。

        返回:
            redis.StrictRedis: 成功连接后返回Redis客户端对象，否则返回None。
        """
        try:
            # 根据配置信息初始化Redis客户端
            self.REDIS = redis.StrictRedis(host=self.config["host"].split(":")[0],
                                     port=int(self.config.get("host", ":6379").split(":")[1]),
                                     db=int(self.config.get("db", 1)),
                                     password=self.config.get("password"),
                                     decode_responses=True)
        except Exception as e:
            # 记录无法连接Redis的警告信息
            logging.warning("Redis can't be connected.")
        return self.REDIS

    def health(self, queue_name):
        """
        检查Redis服务的连通性，并获取指定队列的分组信息。

        :param queue_name: 队列名称
        :return: 返回指定队列的第一个分组信息
        """
        self.REDIS.ping()  # 发送ping命令检查Redis连接是否正常
        return self.REDIS.xinfo_groups(queue_name)[0]  # 获取指定队列的第一个分组信息

    def is_alive(self):
        """
        检查Redis连接是否可用。

        :return: 如果Redis连接存在则返回True，否则返回False
        """
        return self.REDIS is not None

    def exist(self, k):
        """
        检查Redis中是否存在指定的键。

        :param k: 要检查的键
        :return: 如果键存在则返回True，否则返回False
        """
        if not self.REDIS: return
        try:
            return self.REDIS.exists(k)
        except Exception as e:
            logging.warning("[EXCEPTION]exist" + str(k) + "||" + str(e))
            self.__open__()

    def get(self, k):
        """
        从Redis中获取指定键的值。

        :param k: 要获取值的键
        :return: 如果键存在则返回对应的值，否则返回None
        """
        if not self.REDIS: return
        try:
            return self.REDIS.get(k)
        except Exception as e:
            logging.warning("[EXCEPTION]get" + str(k) + "||" + str(e))
            self.__open__()

    def set_obj(self, k, obj, exp=3600):
        """
        将对象存储到Redis中，使用JSON格式序列化对象。

        :param k: 键名
        :param obj: 要存储的对象
        :param exp: 过期时间，默认为3600秒
        :return: 如果存储成功则返回True，否则返回False
        """
        try:
            self.REDIS.set(k, json.dumps(obj, ensure_ascii=False), exp)  # 将对象序列化为JSON字符串并存储到Redis中
            return True
        except Exception as e:
            logging.warning("[EXCEPTION]set_obj" + str(k) + "||" + str(e))
            self.__open__()
        return False

    def set(self, k, v, exp=3600):
        """
        设置键值对并指定过期时间。

        :param k: 键
        :param v: 值
        :param exp: 过期时间，默认为3600秒
        :return: 设置成功返回True，否则返回False
        """
        try:
            self.REDIS.set(k, v, exp)
            return True
        except Exception as e:
            logging.warning("[EXCEPTION]set" + str(k) + "||" + str(e))
            self.__open__()
        return False

    def transaction(self, key, value, exp=3600):
        """
        使用事务设置键值对，确保操作的原子性。

        :param key: 键
        :param value: 值
        :param exp: 过期时间，默认为3600秒
        :return: 设置成功返回True，否则返回False
        """
        try:
            pipeline = self.REDIS.pipeline(transaction=True)
            pipeline.set(key, value, exp, nx=True)
            pipeline.execute()
            return True
        except Exception as e:
            logging.warning("[EXCEPTION]set" + str(key) + "||" + str(e))
            self.__open__()
        return False

    def queue_product(self, queue, message, exp=settings.SVR_QUEUE_RETENTION) -> bool:
        """
        将消息放入队列并设置队列的过期时间。

        :param queue: 队列名称
        :param message: 要发送的消息
        :param exp: 队列的过期时间
        :return: 成功放入队列返回True，否则返回False
        """
        for _ in range(3):
            try:
                payload = {"message": json.dumps(message)}
                pipeline = self.REDIS.pipeline()
                pipeline.xadd(queue, payload)
                pipeline.expire(queue, exp)
                pipeline.execute()
                return True
            except Exception as e:
                print(e)
                logging.warning("[EXCEPTION]producer" + str(queue) + "||" + str(e))
        return False

    def queue_consumer(self, queue_name, group_name, consumer_name, msg_id=b">") -> Payload:
        """
        从指定的Redis流中消费消息。

        该方法首先检查指定的消费组是否存在，如果不存在，则创建它。然后，使用xreadgroup命令从流中读取消息。
        如果没有消息可用，方法将返回None。否则，它将处理消息并返回一个Payload对象。

        参数:
        queue_name (str): 流的名称。
        group_name (str): 消费组的名称。
        consumer_name (str): 消费者的名称。
        msg_id (bytes): 用于读取消息的ID，默认为">"，表示读取最新消息。

        返回:
        Payload: 包含消费的消息的数据结构，如果没有消息可用，则为None。
        """
        try:
            # 获取流的消费组信息。
            group_info = self.REDIS.xinfo_groups(queue_name)
            # 检查指定的消费组是否存在，如果不存在，则创建它。
            if not any(e["name"] == group_name for e in group_info):
                self.REDIS.xgroup_create(
                    queue_name,
                    group_name,
                    id="0",
                    mkstream=True
                )
            # 设置消费参数，包括消费组名、消费者名、消息数量、阻塞时间和消息ID。
            args = {
                "groupname": group_name,
                "consumername": consumer_name,
                "count": 1,
                "block": 10000,
                "streams": {queue_name: msg_id},
            }
            # 从流中读取消息。
            messages = self.REDIS.xreadgroup(**args)
            # 如果没有消息可用，返回None。
            if not messages:
                return None
            # 处理读取到的消息，提取消息ID和负载。
            stream, element_list = messages[0]
            msg_id, payload = element_list[0]
            # 创建并返回Payload对象。
            res = Payload(self.REDIS, queue_name, group_name, msg_id, payload)
            return res
        except Exception as e:
            # 如果异常包含"key"关键字，忽略它，否则记录警告。
            if 'key' in str(e):
                pass
            else:
                logging.warning("[EXCEPTION]consumer" + str(queue_name) + "||" + str(e))
        # 在任何异常或没有消息的情况下，返回None。
        return None


REDIS_CONN = RedisDB()
