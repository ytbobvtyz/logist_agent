#!/usr/bin/env python3
"""
Тестирование RAG с цитатами, источниками и анти-галлюцинациями (Day 24)

Требования:
- Ответ всегда содержит: ответ, список источников, цитаты
- Если релевантность ниже порога → "не знаю" + просьба уточнить

Запуск: python test_llm_rag.py --verify
Требуется: OPENROUTER_API_KEY в .env файле
"""

import os
import sys
import json
import time
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("❌ Ошибка: OPENROUTER_API_KEY не найден")
    sys.exit(1)

from rag_retriever import RAGRetriever


# ============================================================
# 1. КОНФИГУРАЦИЯ
# ============================================================

MODEL = "deepseek/deepseek-v3.2"

# Порог релевантности для ответа "не знаю"
CONFIDENCE_THRESHOLD = 0.25

TEST_QUESTIONS = [
    {"id": 1, "question": "Сколько стоит доставка груза 50 кг из Москвы в Санкт-Петербург у ПЭК?",
     "expected_source": "pecom.txt", "category": "стоимость"},
    {"id": 2, "question": "Какие есть логистические аспекты функционирования транспорта?",
     "expected_source": "transportnaya_logistika-titov_ba.pdf", "category": "образование"},
    {"id": 3, "question": "Расскажи про информационное обеспечение логистики?",
     "expected_source": "transportnaya_logistika-titov_ba.pdf", "category": "образование"},
    {"id": 4, "question": "Какая стоимость доставки ПЭК из Москвы в Казань для груза 100 кг?",
     "expected_source": "pecom.txt", "category": "стоимость"},
    {"id": 5, "question": "Какой URL у публичного API ПЭК для расчёта стоимости?",
     "expected_source": "pecom_api_doc.txt", "category": "техническое"},
    {"id": 6, "question": "Какой максимальный вес посылки у СДЭК?",
     "expected_source": "cdek.txt", "category": "техническое"},
    {"id": 7, "question": "Какой формат ответа возвращает API ПЭК?",
     "expected_source": "pecom_api_doc.txt", "category": "техническое"},
    {"id": 8, "question": "Какие обязанности у фрахтователя?",
     "expected_source": "postanovlenie.txt", "category": "юридическое"},
    {"id": 9, "question": "Что такое фрахтовщик по закону?",
     "expected_source": "postanovlenie.txt", "category": "юридическое"},
    {"id": 10, "question": "Какова максимальная скорость фрахтового корабля в открытом море?",
     "expected_source": None, "category": "нет_информации", "expected_unknown": True},
]


# ============================================================
# 2. ПРОМПТ С ТРЕБОВАНИЕМ ЦИТАТ И ИСТОЧНИКОВ
# ============================================================

def build_prompt_with_sources(query: str, chunks: List[Dict]) -> str:
    """Строит промпт, требующий от модели указать источники и цитаты"""
    
    if not chunks:
        return f"""Ты помощник-логист.

## ВАЖНО: У меня НЕТ информации по этому вопросу в документах.

Ответь строго по шаблону:

## 📋 Ответ
Извините, я не могу ответить на этот вопрос. В предоставленных документах нет информации по теме: "{query}"

## 📚 Источники
Нет источников

## 📝 Цитаты
Нет цитат

## 💡 Рекомендация
Пожалуйста, уточните вопрос или предоставьте дополнительные документы.

Вопрос: {query}"""
    
    # Сортируем чанки по релевантности
    sorted_chunks = sorted(chunks, key=lambda x: x.get('score', 0), reverse=True)
    
    # Формируем контекст с явными маркерами для цитирования
    context_parts = []
    for i, chunk in enumerate(sorted_chunks, 1):
        score = chunk.get('score', 0)
        filename = chunk.get('filename', 'unknown')
        text = chunk.get('text', '')
        
        # Обрезаем слишком длинные чанки
        if len(text) > 800:
            text = text[:800] + "..."
        
        context_parts.append(f"""
--- ИСТОЧНИК [{i}] ---
Файл: {filename}
Релевантность: {score:.3f}
Текст:
{text}
--- КОНЕЦ ИСТОЧНИКА [{i}] ---
""")
    
    context = "\n".join(context_parts)
    
    return f"""Ты помощник-логист.

## ВАЖНЫЕ ПРАВИЛА (НЕ НАРУШАТЬ):

1. **ОБЯЗАТЕЛЬНО** укажи источники в формате: 📄 [имя_файла]
2. **ОБЯЗАТЕЛЬНО** приведи цитаты из документов в формате: 📝 "точная цитата из документа"
3. **ОБЯЗАТЕЛЬНО** структурируй ответ по шаблону ниже
4. Если информация в документах противоречива — укажи это
5. Если релевантность источников низкая (< 0.25) — откажись отвечать

## ФОРМАТ ОТВЕТА (СТРОГО):

## 📋 Ответ
[твой развёрнутый ответ на вопрос]

## 📚 Источники
- 📄 [имя_файла] (релевантность: X.XX)

## 📝 Цитаты
> "точная цитата из документа, подтверждающая ответ"

## ПРИМЕР ПРАВИЛЬНОГО ФОРМАТА:

## 📋 Ответ
Стоимость доставки груза 50 кг из Москвы в Санкт-Петербург у ПЭК составляет 2450 рублей.

## 📚 Источники
- 📄 [pecom.txt]

## 📝 Цитаты
> "Тариф Москва → Санкт-Петербург: До 50 кг: 2450 ₽"

## ТЕПЕРЬ ОТВЕТЬ НА ВОПРОС:

## 📊 Контекст из документов:
{context}

## Вопрос пользователя:
{query}

## Твой ответ (строго по формату с источниками и цитатами):"""


# ============================================================
# 3. АГЕНТ С ЦИТАТАМИ И ИСТОЧНИКАМИ
# ============================================================

class RAGAgentWithCitations:
    def __init__(self, model: str = MODEL, threshold: float = CONFIDENCE_THRESHOLD):
        self.model = model
        self.threshold = threshold
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            timeout=120.0
        )
        self.retriever = None
        
        try:
            self.retriever = RAGRetriever()
            print(f"✅ RAG Retriever загружен")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки RAG: {e}")
    
    def ask_with_citations(self, question: str, top_k: int = 5) -> Dict:
        """
        Запрос к LLM с требованием указать источники и цитаты
        
        Returns:
            Dict с полями: answer, sources, quotes, confidence, chunks
        """
        start_time = time.time()
        
        # Поиск чанков
        chunks = self.retriever.search(question, top_k=top_k) if self.retriever else []
        
        # Проверка релевантности
        avg_score = sum(c.get('score', 0) for c in chunks) / len(chunks) if chunks else 0
        
        # Если релевантность太低, форсируем ответ "не знаю"
        if avg_score < self.threshold or not chunks:
            answer = self._build_unknown_response(question)
            return {
                "answer": answer,
                "sources": [],
                "quotes": [],
                "confidence": avg_score,
                "chunks": chunks,
                "response_time": time.time() - start_time,
                "is_unknown": True,
                "raw_response": answer
            }
        
        # Строим промпт с требованием источников
        prompt = build_prompt_with_sources(question, chunks)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.1
            )
            answer = response.choices[0].message.content or ""
            
            # Парсим ответ для извлечения источников и цитат
            parsed = self._parse_response(answer)
            
            return {
                "answer": parsed["answer"],
                "sources": parsed["sources"],
                "quotes": parsed["quotes"],
                "confidence": avg_score,
                "chunks": chunks,
                "response_time": time.time() - start_time,
                "is_unknown": False,
                "raw_response": answer
            }
            
        except Exception as e:
            return {
                "answer": f"❌ Ошибка: {e}",
                "sources": [],
                "quotes": [],
                "confidence": 0,
                "chunks": chunks,
                "response_time": time.time() - start_time,
                "is_unknown": True,
                "raw_response": ""
            }
    
    def _build_unknown_response(self, question: str) -> str:
        """Строит ответ для случая "не знаю" """
        return f"""## 📋 Ответ
Извините, я не могу уверенно ответить на этот вопрос.

**Причина:** В документах не найдено достаточно релевантной информации по запросу "{question}".

## 📚 Источники
Нет источников (релевантность ниже порога {self.threshold})

## 📝 Цитаты
Нет цитат

## 💡 Рекомендация
Пожалуйста, уточните вопрос или предоставьте дополнительные документы."""
    
    def _parse_response(self, response: str) -> Dict:
        """Парсит ответ для извлечения источников и цитат"""
        sources = []
        quotes = []
        
        # Извлекаем источники (разные форматы)
        source_patterns = [
            r'📄\s*\[([^\]]+)\]',
            r'Источник[:\s]+([^\n]+)',
            r'Файл[:\s]+([^\n]+)',
            r'\[([a-z_]+\.(txt|pdf))\]',
            r'📄\s*([a-z_]+\.(txt|pdf))',
        ]
        
        for pattern in source_patterns:
            found = re.findall(pattern, response, re.IGNORECASE)
            for f in found:
                if isinstance(f, tuple):
                    sources.append(f[0])
                else:
                    sources.append(f)
        
        # Извлекаем цитаты (разные форматы)
        quote_patterns = [
            # Формат с 📝
            r'📝\s*"([^"]+)"',
            r'📝\s*"([^"]+)"',
            # Цитата в кавычках
            r'"([^"]{20,})"',
            r'«([^»]{20,})»',
            # Цитата с маркером >
            r'>\s*"([^"]+)"',
            r'>\s*(.+?)(?=\n\n|\n\[|\n#|$)',
            # Цитата после слова "цитата"
            r'цитат[аы]:\s*"([^"]+)"',
            r'цитат[аы]:\s*(.+?)(?=\n\n|\n\[|\n#|$)',
        ]
        
        for pattern in quote_patterns:
            found = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
            for q in found:
                clean_q = q.strip()
                if len(clean_q) > 20 and clean_q not in quotes:
                    quotes.append(clean_q[:200])
        
        # Если цитат всё ещё нет, пробуем извлечь из секции "Цитаты"
        if not quotes:
            quotes_section = re.search(
                r'##\s*📝\s*Цитаты\s*\n(.*?)(?=\n##|\Z)',
                response,
                re.DOTALL | re.IGNORECASE
            )
            if quotes_section:
                lines = quotes_section.group(1).split('\n')
                for line in lines:
                    # Ищем кавычки
                    quote_match = re.search(r'["«]([^"»]+)["»]', line)
                    if quote_match:
                        quotes.append(quote_match.group(1))
                    # Ищем строки с маркером >
                    elif line.strip().startswith('>'):
                        quote_text = line.strip()[1:].strip()
                        if len(quote_text) > 10:
                            quotes.append(quote_text)
        
        # Извлекаем основной ответ
        answer_match = re.search(
            r'##\s*📋\s*Ответ\s*\n(.*?)(?=\n##\s*📚\s*Источники|\n##\s*📝\s*Цитаты|\Z)',
            response,
            re.DOTALL | re.IGNORECASE
        )
        answer = answer_match.group(1).strip() if answer_match else response[:500]
        
        # Очищаем источники от дубликатов и пустых строк
        sources = [s for s in sources if s and len(s) > 2]
        sources = list(dict.fromkeys(sources))
        
        # Очищаем цитаты от дубликатов
        quotes = list(dict.fromkeys(quotes))
        
        return {
            "answer": answer,
            "sources": sources[:5],
            "quotes": quotes[:3]
        }


# ============================================================
# 4. ВЕРИФИКАЦИЯ КАЧЕСТВА
# ============================================================

def verify_response(result: Dict, test: Dict) -> Dict:
    """Проверяет ответ на соответствие требованиям Day 24"""
    
    checks = {
        "has_answer": bool(result.get('answer') and len(result['answer']) > 20),
        "has_sources": len(result.get('sources', [])) > 0,
        "has_quotes": len(result.get('quotes', [])) > 0,
        "correct_unknown": False,
        "sources_match_chunks": False,
    }
    
    # Проверка для вопросов без информации
    if test.get('expected_unknown', False):
        checks['correct_unknown'] = result.get('is_unknown', False) or "не могу" in result.get('answer', '').lower()
    
    # Проверка соответствия источников
    if result.get('chunks') and result.get('sources'):
        chunk_filenames = [c.get('filename', '') for c in result['chunks']]
        for source in result['sources']:
            if any(source in cf for cf in chunk_filenames):
                checks['sources_match_chunks'] = True
                break
    
    score = sum(checks.values()) / len(checks) * 100
    
    return {
        "checks": checks,
        "score": round(score, 1),
        "passed": score >= 60
    }


# ============================================================
# 5. ОСНОВНАЯ ФУНКЦИЯ
# ============================================================

def run_verification():
    print("="*80)
    print("🔍 ДЕНЬ 24: ВЕРИФИКАЦИЯ ЦИТАТ, ИСТОЧНИКОВ И АНТИ-ГАЛЛЮЦИНАЦИЙ")
    print("="*80)
    print(f"📅 Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🤖 Модель: {MODEL}")
    print(f"⚙️  Порог уверенности: {CONFIDENCE_THRESHOLD}")
    print("="*80)
    
    agent = RAGAgentWithCitations(threshold=CONFIDENCE_THRESHOLD)
    
    results = []
    total_score = 0
    
    for test in TEST_QUESTIONS:
        print(f"\n{'─'*80}")
        print(f"📌 ВОПРОС {test['id']}: {test['question']}")
        print(f"   Категория: {test['category']}")
        if test.get('expected_unknown'):
            print(f"   🎯 Ожидается ответ 'не знаю'")
        print("-"*40)
        
        # Запрос к агенту
        result = agent.ask_with_citations(test['question'])
        
        # Верификация
        verification = verify_response(result, test)
        
        print(f"\n  ⏱️  Время: {result['response_time']:.2f}с")
        print(f"  📊 Уверенность: {result['confidence']:.3f}")
        print(f"  📚 Источников найдено: {len(result['sources'])}")
        print(f"  📝 Цитат найдено: {len(result['quotes'])}")
        
        print(f"\n  📋 Ответ:")
        print(f"     {result['answer'][:300]}...")
        
        if result['sources']:
            print(f"\n  📚 Источники:")
            for s in result['sources'][:3]:
                print(f"     • {s}")
        
        if result['quotes']:
            print(f"\n  📝 Цитаты:")
            for q in result['quotes'][:2]:
                print(f"     • \"{q[:100]}...\"")
        
        print(f"\n  ✅ Верификация:")
        for check, passed in verification['checks'].items():
            status = "✅" if passed else "❌"
            print(f"     {status} {check}: {passed}")
        
        print(f"\n  📊 Оценка: {verification['score']}%")
        
        results.append({
            "id": test['id'],
            "question": test['question'],
            "category": test['category'],
            "expected_unknown": test.get('expected_unknown', False),
            "verification": verification,
            "sources": result['sources'],
            "quotes": result['quotes'],
            "confidence": result['confidence'],
            "response_time": result['response_time'],
            "answer_preview": result['answer'][:200]
        })
        
        total_score += verification['score']
        
        time.sleep(1)
    
    # ===== ИТОГОВАЯ СТАТИСТИКА =====
    print("\n" + "="*80)
    print("📊 ИТОГОВАЯ СТАТИСТИКА")
    print("="*80)
    
    avg_score = total_score / len(results)
    
    # Подсчёт метрик
    has_sources_count = sum(1 for r in results if len(r['sources']) > 0)
    has_quotes_count = sum(1 for r in results if len(r['quotes']) > 0)
    unknown_correct = sum(1 for r in results if r['expected_unknown'] and r['verification']['checks']['correct_unknown'])
    unknown_total = sum(1 for r in results if r['expected_unknown'])
    
    print(f"\n📈 ОБЩАЯ ОЦЕНКА: {avg_score:.1f}%")
    
    print(f"\n📊 ДЕТАЛИЗАЦИЯ:")
    print(f"   • Ответы с источниками:    {has_sources_count}/{len(results)} ({has_sources_count/len(results)*100:.0f}%)")
    print(f"   • Ответы с цитатами:       {has_quotes_count}/{len(results)} ({has_quotes_count/len(results)*100:.0f}%)")
    if unknown_total > 0:
        print(f"   • Корректные 'не знаю':    {unknown_correct}/{unknown_total} ({unknown_correct/unknown_total*100:.0f}%)")
    
    print(f"\n📋 ДЕТАЛЬНАЯ ТАБЛИЦА:")
    print(f"{'ID':<4} {'Категория':<14} {'Оценка':<8} {'Источники':<10} {'Цитаты':<8} {'Уверенность':<12}")
    print("-"*80)
    
    for r in results:
        sources_count = len(r['sources'])
        quotes_count = len(r['quotes'])
        print(f"{r['id']:<4} {r['category']:<14} {r['verification']['score']:>5.1f}%    {sources_count:<8} {quotes_count:<8}   {r['confidence']:.3f}")
    
    # Сохранение результатов
    output = {
        "test_date": datetime.now().isoformat(),
        "model": MODEL,
        "threshold": CONFIDENCE_THRESHOLD,
        "summary": {
            "total_questions": len(results),
            "avg_score": round(avg_score, 1),
            "has_sources_count": has_sources_count,
            "has_quotes_count": has_quotes_count,
            "unknown_correct": unknown_correct,
            "unknown_total": unknown_total
        },
        "results": results
    }
    
    filename = f"citations_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Результаты сохранены: {filename}")
    
    # Финальный вердикт
    print("\n" + "="*80)
    print("🎯 ВЕРДИКТ:")
    
    if avg_score >= 80:
        print("   ✅ ОТЛИЧНО! Агент корректно указывает источники и цитаты")
        print("   📚 Анти-галлюцинации работают")
    elif avg_score >= 60:
        print("   ⚠️ ХОРОШО, но есть куда расти")
        print("   🔧 Рекомендуется улучшить извлечение цитат")
    else:
        print("   ❌ ТРЕБУЕТ ДОРАБОТКИ")
        print("   🔧 Проверьте качество чанков и промпт")
    
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", default=True)
    args = parser.parse_args()
    
    run_verification()