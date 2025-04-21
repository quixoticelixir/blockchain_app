from gigachat import GigaChat

GIGACHAT_SCOPE = "GIGACHAT_API_PERS"

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
        with GigaChat(credentials=auth_basic_value, verify_ssl_certs=False) as giga:
             response = giga.chat(prompt)

             if response and response.choices and len(response.choices) > 0 and response.choices[0].message and response.choices[0].message.content:
                 description = response.choices[0].message.content
             else:
                 print(f"Warning: GigaChat SDK chat response did not contain expected text content structure. Response: {response}")
                 return None


    except Exception as e:
        print(f"Error during GigaChat SDK call: {e}")
        raise e

    return description
