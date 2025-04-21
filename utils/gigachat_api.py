# utils/gigachat_api.py
# Удаляем requests, json, uuid, base64
# import requests
# import json
# import uuid
# import base64

# Импортируем класс GigaChat из SDK
from gigachat import GigaChat # type: ignore # Добавим type: ignore если ваш линтер ругается до установки

# --- Константы API (могут больше не понадобиться явно, но оставим scope) ---
# GIGACHAT_AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth" # Не нужен
# GIGACHAT_CHAT_URL = "https://gigachat.api.sbercloud.ru/rest/v1/chat/completions" # Не нужен
GIGACHAT_SCOPE = "GIGACHAT_API_PERS" # Scope может понадобиться при инициализации SDK

# --- Функция для получения описания с помощью GigaChat SDK ---

# Функция теперь принимает готовую base64 строку авторизации
def get_ai_description_from_stats(auth_basic_value: str, stats_text: str) -> str | None:
    """
    Получает описание кластеров от GigaChat с помощью SDK.
    auth_basic_value: строка, которая является результатом base64(Client ID:Client Secret).
    stats_text: отформатированная статистика кластеров (например, в Markdown).
    Возвращает текст описания (строка) в случае успеха.
    Возвращает None, если SDK не вернул ожидаемый текст.
    Вызывает исключения при ошибках (например, ошибки аутентификации, сетевые).
    """

    # Ваш промпт для GigaChat
    prompt = f"""
Проанализируй представленную ниже статистику по кластерам кошельков.
Статистика содержит описательные метрики (среднее, стандартное отклонение, минимумы, максимумы, квантили)
для различных признаков активности (баланс, количество входящих/исходящих транзакций, объемы, уникальные контрагенты, активные дни).

Для каждого кластера:
1. Кратко опиши его ключевые характеристики на основе статистик.
2. Предположи, к какой категории пользователей мог бы относиться этот кластер (например, 'Киты', 'Арбитражники', 'Пассивные пользователи', 'Мелкие трейдеры', 'Активные пользователи', 'Новички' и т.п.), объясни почему.

Представь результат в виде структурированного текста, описывая каждый кластер отдельно.

Статистика по кластерам:
{stats_text}
"""
    description = None
    try:
        # Инициализируем SDK.
        # credentials=auth_basic_value передает вашу base64 строку для получения токена.
        # verify_ssl_certs=False может быть нужен для тестовых стендов, как и раньше.
        # scope=GIGACHAT_SCOPE возможно потребуется, проверьте документацию SDK.
        with GigaChat(credentials=auth_basic_value, verify_ssl_certs=False) as giga:
             # Выполняем запрос к модели. SDK сам заботится о токене.
             response = giga.chat(prompt)

             # Извлекаем текст ответа. Структура ответа может отличаться от ручной обработки.
             # Проверьте структуру от объекта response от SDK.
             # Обычно текст находится в response.choices[0].message.content
             if response and response.choices and len(response.choices) > 0 and response.choices[0].message and response.choices[0].message.content:
                 description = response.choices[0].message.content
             else:
                 print(f"Warning: GigaChat SDK chat response did not contain expected text content structure. Response: {response}")
                 return None # Возвращаем None, если нет ожидаемого контента


    except Exception as e:
        # SDK может выбрасывать различные исключения (AuthError, APIError, ConnectionError и т.д.)
        print(f"Error during GigaChat SDK call: {e}")
        # В SDK нет Response Body в таком виде, но можно попытаться вывести объект ошибки
        # print(f"Error details: {e.detail}" if hasattr(e, 'detail') else '') # Пример, зависит от SDK
        raise e # Перевыбрасываем исключение для обработки в Streamlit

    return description

# Функции get_gigachat_token и get_gigachat_description УДАЛЕНЫ из этого файла
# т.к. их логика теперь внутри get_ai_description_from_stats и SDK