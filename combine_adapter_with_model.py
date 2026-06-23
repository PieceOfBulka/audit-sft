import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def parse_args():
    p = argparse.ArgumentParser(description='Combine model and adapter into one LM')
    p.add_argument(
        '--model',
        help='path to base model'
    )
    p.add_argument(
        '--adapter',
        help='path to adapter'
    )
    p.add_argument(
        '--result',
        default=None,
        help='path to result model'
    )
    return p.parse_args()


def main():
    args = parse_args()
    
    print(f'Загрузка базовой модели: {args.model}')
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map='cpu'
    )
    print(f'Добавление адаптера: {args.adapter}')
    model = PeftModel.from_pretrained(base_model, args.adapter)

    print(f'Сохранение объединенной модели в {args.result}')
    save_path = args.result if args.result else f'weights/combined_{args.model.split('/')[-1]}'
    try:
        model.merge_and_unload().save_pretrained(args.result)
        AutoTokenizer.from_pretrained(args.model).save_pretrained(args.result)
        print(f'Новая модель успешно сохранена!')
    except Exception as e:
        print(f'Возникла ошибка при сохранении: {e}')


if __name__=='__main__':
    main()