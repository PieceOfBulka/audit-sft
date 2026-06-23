import argparse
import os
import sys
import torch
import json
from dotenv import load_dotenv

from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM

from lora_peft.sft_lora_peft import pick_device, torch_dtype

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_TOKEN'))

_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ADAPTER = os.path.join(_ROOT, "lora_peft", "lora-adapter")

# Защитный потолок на суммарную длину ответа оцениваемой модели (догенерация до EOS).
MAX_TOTAL_NEW_TOKENS = 8192

DEFAULT_SYSTEM = (
    "Ты — эксперт по внутреннему аудиту и управлению рисками. "
    "Отвечай в профессиональном регистре, используя корректную "
    "терминологию, структурируя рассуждение в логике системы "
    "управления рисками. Отвечай на русском."
)

QUESTION_PROMPT = (
    "Ты — опытный, дотошный и требовательный внешний аудитор или регулятор. "
    "Твоя цель — составить ОДИН сложный, глубокий или каверзный вопрос для проверки знаний "
    "внутреннего аудитора компании по предоставленному регламенту/документу.\n\n"
    "ПРАВИЛА ГЕНЕРАЦИИ:\n"
    "1. Избегай банальных вопросов в лоб (например: 'Что написано в пункте 2?').\n"
    "2. Используй один из следующих подходов для усложнения теста:\n"
    "   - Ситуационный кейс (Case Study): опиши реалистичную проблему на производстве/в финансах и спроси, как действовать согласно регламенту.\n"
    "   - Поиск противоречий: спроси о действиях в пограничной ситуации, где правила регламента могут пересекаться.\n"
    "   - Проверка на 'серые зоны': задай вопрос о зонах ответственности, матрице рисков или порядке эскалации нарушений.\n"
    "3. Не пиши никакого введения, приветствий и пояснений. На выходе должен быть ТОЛЬКО сам текст вопроса."
)

JUDGE_PROMPT = (
    "Ты — главный методолог службы внутреннего аудита и комплаенса крупной корпорации. "
    "Твоя задача — беспристрастно и строго оценить ответ тестируемого аудитора.\n\n"
    "Тебе будут предоставлены вопрос и ответ тестируемого\n\n"
    "КРИТЕРИИ ОЦЕНКИ (по шкале от 1 до 5):\n"
    "1. Обоснованность (Faithfulness): Насколько ответ соответствует исходному регламенту. "
    "Если в ответе есть выдуманные факты, искажения процедур, ложные ссылки на законы или галлюцинации — снижай балл. "
    "Оценка 5 ставится только если ВСЕ факты в ответе подтверждаются текстом регламента.\n"
    "2. Полнота (Completeness): Насколько исчерпывающе дан ответ на заданный вопрос. "
    "Учтены ли все важные риски, шаги или условия, упомянутые в вопросе и регламенте?\n\n"
    "ФОРМАТ ВЫВОДА:\n"
    "Ты должен вернуть ответ строго в формате JSON, без какого-либо другого текста вокруг или markdown-разметки (без ```json). "
    "JSON должен содержать три поля:\n"
    "{\n"
    "  \"faithfulness_score\": 5,\n"
    "  \"completeness_score\": 4\n"
    "  \"reasoning\": \"Подробный разбор ответа на русском языке. Перечисли плюсы, минусы, неточности или пропущенные важные детали регламента.\",\n"
    "}"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='LLM as a judge оценка')
    p.add_argument(
        '--model',
        default='',
        help='Путь к модели, которую хотим оценить'
    )
    p.add_argument(
        '--lora',
        action='store_true',
        help='Загрузить LoRA-адаптер поверх базовой модели'
    )
    p.add_argument(
        '--adapter',
        default=DEFAULT_ADAPTER,
        help='Загрузить LoRA-адаптер поверх базовой модели'
    )
    p.add_argument(
        "--system",
        default=DEFAULT_SYSTEM,
        help="Системный промпт (можно сменить в чате командой /system)",
    )
    p.add_argument("--max-new-tokens", type=int, default=1024,
                   help="Размер чанка генерации; ответ догенерируется до конца (EOS), не обрезаясь")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument(
        "--greedy",
        action="store_true",
        help="Детерминированная генерация (без сэмплинга)",
    )
    return p.parse_args()

def load_model(use_lora: bool, model_path: str, adapter_path: str, device: str):
    if not os.path.isdir(model_path):
            print(f"Ошибка: модель не найдена: {model_path}", file=sys.stderr)
            sys.exit(1)
    print("Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    print("Загрузка базовой модели (это может занять минуту)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch_dtype,
        low_cpu_mem_usage=True
    )

    if use_lora:
        from peft import PeftModel

        if not os.path.isdir(adapter_path):
            print(f"Ошибка: адаптер не найден: {adapter_path}", file=sys.stderr)
            sys.exit(1)
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"LoRA-адаптер загружен: {adapter_path}")
    else:
        print("Режим: базовая модель без LoRA")

    model.to(device)
    model.eval()
    return model, tokenizer

@torch.no_grad()
def generate_reply(
    model,
    tokenizer,
    question: str,
    system_prompt: str,
    device: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    greedy: bool,
) -> str:
    tokenized_prompt = tokenizer.apply_chat_template(
        [
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': question
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        tokenized_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)

    gen_kwargs: dict = {
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if greedy:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs.update(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    # Полный ответ без обрезки: чанки по max_new_tokens до EOS либо до защитного потолка.
    seq = inputs["input_ids"]
    attn = inputs["attention_mask"]
    new_ids: list[int] = []
    while len(new_ids) < MAX_TOTAL_NEW_TOKENS:
        budget = min(max_new_tokens, MAX_TOTAL_NEW_TOKENS - len(new_ids))
        out = model.generate(
            input_ids=seq, attention_mask=attn, max_new_tokens=budget, **gen_kwargs
        )
        gen = out[0][seq.shape[1]:]
        new_ids.extend(gen.tolist())
        if gen.numel() == 0 or gen[-1].item() == tokenizer.eos_token_id or gen.numel() < budget:
            break
        seq = out
        attn = torch.ones_like(seq)
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def generate_question():
    question_generation = client.chat.completions.create(
        model=os.getenv('QUESTION_MODEL_NAME','gpt-5-nano'),
        messages=[
            {
                'role': 'system',
                'content': QUESTION_PROMPT
            }
        ]
    )

    return question_generation.choices[0].message.content

def generate_judgement(question: str, reply: str):
    judgement_generation = client.chat.completions.create(
        model=os.getenv('JUDGE_MODEL_NAME','gpt-5-nano'),
        messages=[
            {
                'role': 'user',
                'content': question
            },
            {
                'role': 'assistant',
                'content': reply
            },
            {
                'role': 'system',
                'content': JUDGE_PROMPT
            }
        ],
        response_format={ "type": "json_object" }
    )

    return judgement_generation.choices[0].message.content



def main():
    args = parse_args()
    device = pick_device()
    print(f"Устройство: {device}")

    reply_model, reply_tokenizer = load_model(
        use_lora=args.lora,
        model_path=args.model,
        adapter_path=args.adapter,
        device=device
        )

    print('===Начало цикла llm-as-judge\n')
    while True:
        question = generate_question()
        print(f'=Вопрос:\n{question}')
        reply = generate_reply(
            model=reply_model,
            tokenizer=reply_tokenizer,
            question=question,
            system_prompt=args.system,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            greedy=args.greedy)
        print(f'=Ответ модели:\n{reply}')
        judge = generate_judgement(question, reply)
        judge_dict = json.loads(judge)
        reasoning = judge_dict.get("reasoning", "Описание отсутствует")
        faithfulness = judge_dict.get("faithfulness_score", 0)
        completeness = judge_dict.get("completeness_score", 0)
        print(f'=Оценка судьи:\n{json.dumps(judge_dict,indent=2,ensure_ascii=False)}')

        result_file = f'judgements/{args.model.split('/')[-1]}'
        if args.lora:
            result_file += f'&&{args.adapter.split('/')[-1]}'
        result_file += '.txt'
        with open(result_file, 'a', encoding='utf-8') as f:
            f.write('\n==============')
            f.write(f'\n\n==Question\n{question}')
            f.write(f'\n\n==Reply\n{reply}')
            f.write(f'\n\n==Judgement\n')
            f.write(f'- faithfulness = {faithfulness}\n')
            f.write(f'- completeness = {completeness}\n')
            f.write(f'- Reasoning\n{reasoning}\n')

        is_continue = input('\n====Продолжаем? (y/n): ')
        if is_continue!='y':
            break

if __name__=='__main__':
    main()