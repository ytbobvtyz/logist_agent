"""
Утилиты для работы с асинхронным кодом.
Управление event loop, запуск корутин в фоновых потоках.
"""

import asyncio
import threading
from typing import Any, Callable, Optional, List
import sys
from concurrent.futures import Future
from functools import partial


class BackgroundEventLoop:
    """Фоновый event loop в отдельном потоке."""
    
    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._started = False
    
    def start(self) -> None:
        """Запускает фоновый event loop."""
        if self._started:
            return
        
        self._loop = asyncio.new_event_loop()
        
        def _run_loop(loop: asyncio.AbstractEventLoop):
            asyncio.set_event_loop(loop)
            loop.run_forever()
        
        self._thread = threading.Thread(
            target=_run_loop,
            args=(self._loop,),
            daemon=True,
            name="BackgroundEventLoop"
        )
        self._thread.start()
        self._started = True
        
        print("✅ Фоновый event loop запущен")
    
    def stop(self) -> None:
        """Останавливает фоновый event loop."""
        if not self._started or not self._loop:
            return
        
        self._loop.call_soon_threadsafe(self._loop.stop)
        
        if self._thread:
            self._thread.join(timeout=5)
        
        self._started = False
        self._loop = None
        self._thread = None
        
        print("✅ Фоновый event loop остановлен")
    
    def run_async(self, coro) -> Any:
        """
        Запускает корутину на фоновом loop'е и возвращает результат.
        
        Args:
            coro: Корутина для выполнения
            
        Returns:
            Результат выполнения корутины
            
        Raises:
            RuntimeError: Если loop не запущен
            TimeoutError: Если выполнение превысило таймаут
        """
        if not self._started or not self._loop:
            raise RuntimeError("Фоновый event loop не запущен")
        
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=120)
    
    @property
    def loop(self) -> Optional[asyncio.AbstractEventLoop]:
        """Возвращает фоновый event loop."""
        return self._loop
    
    @property
    def is_running(self) -> bool:
        """Проверяет, запущен ли event loop."""
        return self._started


# Глобальный экземпляр фонового event loop
_bg_loop = BackgroundEventLoop()


def get_background_loop() -> BackgroundEventLoop:
    """Возвращает глобальный фоновый event loop."""
    global _bg_loop
    return _bg_loop


def start_background_loop() -> None:
    """Запускает глобальный фоновый event loop."""
    global _bg_loop
    _bg_loop.start()


def stop_background_loop() -> None:
    """Останавливает глобальный фоновый event loop."""
    global _bg_loop
    _bg_loop.stop()


def run_in_background(coro) -> Any:
    """
    Запускает корутину в фоновом event loop'е.
    
    Args:
        coro: Корутина для выполнения
        
    Returns:
        Результат выполнения корутины
    """
    global _bg_loop
    return _bg_loop.run_async(coro)


def sync_to_async(func: Callable) -> Callable:
    """
    Декоратор для преобразования синхронной функции в асинхронную.
    
    Args:
        func: Синхронная функция
        
    Returns:
        Асинхронная функция
    """
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
    
    return wrapper


async def run_with_timeout(coro, timeout: float = 30.0) -> Any:
    """
    Выполняет корутину с таймаутом.
    
    Args:
        coro: Корутина для выполнения
        timeout: Таймаут в секундах
        
    Returns:
        Результат выполнения корутины
        
    Raises:
        TimeoutError: Если выполнение превысило таймаут
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError(f"Операция превысила таймаут {timeout} секунд")


class AsyncContextManager:
    """Базовый класс для асинхронных контекстных менеджеров."""
    
    async def __aenter__(self):
        await self.setup()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
    
    async def setup(self):
        """Настройка ресурсов."""
        pass
    
    async def cleanup(self):
        """Очистка ресурсов."""
        pass


# Global async helper instance
async_helper = AsyncHelper()