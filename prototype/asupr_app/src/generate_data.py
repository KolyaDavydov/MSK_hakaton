import pandas as pd
from datetime import datetime, timedelta, time
import time
import numpy as np
import random
import clickhouse_connect
from tqdm import tqdm
import zoneinfo
import sys
import warnings
warnings.filterwarnings('ignore')




def generate_data(df_origin, first_day='2025-01-01', last_day='2025-12-07', id=0):
    df = df_origin.copy()
    df['day'] = df['datetime'].dt.day
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['dayofweek'] = df['dayofweek'].apply(lambda x: 0 if x <= 3 else 1)

    start_date = datetime.strptime(first_day, '%Y-%m-%d')
    end_date = datetime.strptime(last_day, '%Y-%m-%d')
    
    date_list = []
    current_date = start_date
    
    while current_date <= end_date:
        date_list.append(current_date)
        current_date += timedelta(days=1)

    temp_list = []
    for day in date_list:
        weekend = 1 if day.weekday() >= 4 else 0
        temp = df[df['dayofweek'] == weekend]

        random_day = random.choice(temp['day'].unique())
        temp = temp[temp['day'] == random_day]
        temp['time'] = temp['datetime'].dt.time
        temp['datetime'] = temp['time'].apply(lambda t: datetime.combine(day.date(), t))
       
        
        temp = temp[['datetime', 'input_hot', 'output_hot', 'rashod_hot', 'temp_input', 'temp_output', 'cumulative_rashod_cold', 'rashod_cold']]
        temp_list.append(temp)

    total_df = pd.concat(temp_list, axis=0)
    total_df.insert(0, 'id', id)



    total_df = total_df.reset_index(drop=True)

    my_list = [
        random.uniform(0.01,0.1),
        random.uniform(0.2,0.3),
        random.uniform(0.4,0.5),
        random.uniform(0.8,0.95),
        ]
    random_count = np.random.randint(2, 5)


    # Создаем веса: у последнего элемента вес в 5 раз меньше (80% снижение)
    weights = [1, 1, 1, 0.02]  # Последний элемент имеет вес 0.2 (80% снижение)

    # Нормализуем веса
    normalized_weights = np.array(weights) / np.sum(weights)

    random_elements = np.random.choice(my_list, size=random_count, replace=False, p=normalized_weights)

    for i in random_elements:

        # Определяем диапазон строк
        decrease_start = int(len(total_df) * i)
        decrease_end = decrease_start + random.randint(100, 270)
        zero_start = decrease_end +1
        zero_end = zero_start + random.randint(10,100)

        num_decrease_rows = decrease_end - decrease_start + 1
        num_zero_rows = zero_end - zero_start + 1

        # 1. Постепенное уменьшение с 300 по 564 строку
        for col in ['rashod_hot', 'input_hot', 'output_hot']:
            original_values = total_df.loc[decrease_start:decrease_end, col].values
            decrease_factors = np.linspace(1, 0, num_decrease_rows)
            total_df.loc[decrease_start:decrease_end, col] = original_values * decrease_factors

        # 2. Устанавливаем нули с 565 по 664 строку
        total_df.loc[zero_start:zero_end, ['rashod_hot', 'input_hot', 'output_hot']] = 0

        # Для temp_input - уменьшаем существующие значения на 10 градусов
        original_temp_input = total_df.loc[decrease_start:zero_end, 'temp_input'].values
        decrease_temp_input = np.linspace(0, 10, num_decrease_rows + num_zero_rows)
        total_df.loc[decrease_start:zero_end, 'temp_input'] = original_temp_input - decrease_temp_input

        # Для temp_output - уменьшаем существующие значения на 14 градусов
        original_temp_output = total_df.loc[decrease_start:zero_end, 'temp_output'].values
        decrease_temp_output = np.linspace(0, 14, num_decrease_rows + num_zero_rows)
        total_df.loc[decrease_start:zero_end, 'temp_output'] = original_temp_output - decrease_temp_output

    random_multiplier = random.uniform(1,10)
    total_df['input_hot'] = (total_df['input_hot'] * random_multiplier).round(2)
    total_df['output_hot'] = (total_df['output_hot'] * random_multiplier).round(2)
    total_df['rashod_hot'] = (total_df['rashod_hot'] * random_multiplier).round(2)
    total_df['rashod_cold'] = total_df['rashod_cold'] * random_multiplier
    total_df['cumulative_rashod_cold'] = total_df['rashod_cold'].cumsum().shift(1, fill_value=0) + np.random.uniform(5678, 15046)
    total_df['rashod_cold'] = total_df['rashod_cold'].round(2)
    total_df['cumulative_rashod_cold'] = total_df['cumulative_rashod_cold'].round(3)
    total_df['temp_output'] = total_df['temp_output'].astype(int)
    total_df['temp_input'] = total_df['temp_input'].astype(int)
    
    return total_df



df_hot= pd.read_excel('/app/src/GVS.xlsx')
df_hot['Дата'] = pd.to_datetime(df_hot['Дата'], format='%d.%m.%Y', errors='coerce')
df_hot['datetime'] = df_hot['Дата'] + pd.to_timedelta(df_hot['Время суток, ч'].str.split('-').str[0].astype(int), unit='h')
df_hot.drop(['Дата', 'Время суток, ч'], axis=1, inplace=True)
df_hot.rename(columns={
    'Подача, м3': 'input_hot',
    'Обратка, м3': 'output_hot',
    'Потребление за период, м3': 'rashod_hot',
    'Т1 гвс, оС':'temp_input',
    'Т2 гвс, оС': 'temp_output'
    }, inplace=True)


df_cold= pd.read_excel('/app/src/HVS.xlsx')
df_cold['Дата'] = pd.to_datetime(df_cold['Дата'], format='%d.%m.%Y', errors='coerce')
df_cold['datetime'] = df_cold['Дата'] + pd.to_timedelta(df_cold['Время суток, ч'].str.split('-').str[0].astype(int), unit='h')
df_cold.drop(['Дата', 'Время суток, ч'], axis=1, inplace=True)
df_cold.rename(columns={
    'Потребление накопленным итогом, м3': 'cumulative_rashod_cold',
    'Потребление за период, м3': 'rashod_cold'
    }, inplace=True)



df = df_hot.merge(df_cold, on='datetime')


list_pandas = []
for i in tqdm(range(1, 81), file=sys.stdout):
    temp = generate_data(df[df['datetime'] < datetime(2025, 4, 17)], first_day='2024-10-14', last_day='2025-10-01', id=i)
    list_pandas.append(temp)
    # temp.to_csv(f"{i}.csv", index=False)

total_df = pd.concat(list_pandas, axis=0)


# Создание клиента
client = clickhouse_connect.get_client(
    host='clickhouse',
    port=8123,
    username='admin',
    password='password',
)

client.command('DROP TABLE IF EXISTS msk_database.analytic')
# Создание базы данных
client.command('CREATE DATABASE IF NOT EXISTS msk_database')

create_table_query = '''
CREATE TABLE IF NOT EXISTS msk_database.analytic (
    id Int32,
    datetime DateTime,
    input_hot Float32,
    output_hot Float32,
    rashod_hot Float32,
    temp_input Int32,
    temp_output Int32,
    cumulative_rashod_cold Float32,
    rashod_cold Float32
) ENGINE = MergeTree()
ORDER BY (datetime)
'''

client.command(create_table_query)

client.insert_df('msk_database.analytic', total_df)


while True:
    # Текущее время округленное до часов вниз
    current_hour = pd.Timestamp.now(zoneinfo.ZoneInfo('Europe/Moscow')).replace(tzinfo=None).floor('H')

    click_df = client.query_df(
        f"""
        SELECT *
        FROM msk_database.analytic
        ORDER BY datetime DESC
        LIMIT {total_df['id'].nunique() * 24 * 77}
        """)


    if click_df['datetime'].max() == current_hour:
        # print(f"Максимальное время в таблице {click_df['datetime'].max()}")
        time.sleep(10)
        continue
    else:
        list_df = []
        for id in click_df['id'].unique():
            # temp = click_df[(click_df['id'] == id) & (click_df['datetime'] == click_df['datetime'].max() - pd.Timedelta(days=6, hours=23))]
            temp = click_df[(click_df['id'] == id) & (click_df['datetime'] == click_df['datetime'].max() - pd.Timedelta(days=69, hours=23))]
            list_df.append(temp)
        
        new_df = pd.concat(list_df, axis=0)
        new_df['datetime'] = new_df['datetime'] + pd.Timedelta(weeks=10)
        new_df['cumulative_rashod_cold'] = new_df['cumulative_rashod_cold'] + new_df['rashod_cold']
        new_df = new_df.reset_index(drop=True)
        client.insert_df('msk_database.analytic', new_df)
        print(new_df.tail(1))
        time.sleep(0.01)
        # print(f"Максимальное время в НОВОЙ таблице {new_df['datetime'].max()}")