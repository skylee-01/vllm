import asyncio
import time

async def fetch_data(url):
    """
    模拟一个异步的网络请求。
    """
    print(f"开始从 {url} 获取数据...")
    await asyncio.sleep(2)  # 模拟网络延迟 2 秒
    print(f"从 {url} 获取数据完成。")
    return f"来自 {url} 的数据"

async def main():
    """
    主协程函数，协调多个异步任务。
    """
    start_time = time.time()
    print("主程序开始运行...")

    # 同时启动两个异步任务
    task1 = asyncio.create_task(fetch_data("http://example.com/api/data1"))
    task2 = asyncio.create_task(fetch_data("http://example.com/api/data2"))

    # 等待这两个任务都完成
    result1 = await task1
    result2 = await task2

    print(f"收到结果 1: {result1}")
    print(f"收到结果 2: {result2}")

    end_time = time.time()
    print(f"主程序运行结束，总耗时: {end_time - start_time:.2f} 秒。")

# 运行主协程
if __name__ == "__main__":
    asyncio.run(main())