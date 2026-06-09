import json
import os

dataset=[]

for file in os.listdir('data/questions'):
    print(f'Начало обработки файла {file}')
    try:
        with open(f'data/questions/{file}', 'r', encoding='utf-8') as f:
            df=json.load(f)
        print(f'В файле найдено {len(df)} записей')

        for element in df:
            dataset.append(element)
    except Exception as e:
        print(f'Ошибка = {e}')
    print()

print(f'Всего записей в датасете = {len(dataset)}')

with open(f'data/dataset_questions.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)