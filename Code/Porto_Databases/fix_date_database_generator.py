import os
import csv
import ast
import random
import math
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def adjust_to_fixed_date(timestamp):
    """
    Ajusta un timestamp para que la fecha sea siempre 1 de febrero de 2014,
    conservando únicamente la hora y el minuto originales.
    """
    original_time = datetime.fromtimestamp(timestamp)
    fixed_date = datetime(2014, 2, 1, original_time.hour, original_time.minute, original_time.second)
    return int(fixed_date.timestamp())

def select_trajectories_exponential(
    trajectory_file: str,
    min_length: int,
    max_length: int,
    N: int,
    lambda_: float,
    output_file: str
) -> None:
    if not os.path.exists(trajectory_file):
        raise FileNotFoundError(f"El archivo {trajectory_file} no existe.")

    # Leer las trayectorias del archivo
    with open(trajectory_file, "r") as infile:
        reader = csv.reader(infile)
        trajectories = [
            (int(row[0]), ast.literal_eval(row[1]))  # Guardar el timestamp inicial y la trayectoria
            for row in reader
        ]
    
    # Filtrar las trayectorias según las longitudes válidas y el intervalo de tiempo
    filtered_trajectories = [
        (timestamp, trajectory) for timestamp, trajectory in trajectories
        if min_length <= len(trajectory) <= max_length
    ]

    if not filtered_trajectories:
        raise ValueError("No hay trayectorias que cumplan con los criterios especificados.")

    # Calcular las probabilidades según una distribución exponencial invertida
    probabilities = [
        math.exp(-lambda_ * (max_length - length)) for length in range(min_length, max_length + 1)
    ]
    probabilities = [p / sum(probabilities) for p in probabilities]  # Normalizar

    # Calcular las frecuencias deseadas para cada longitud
    target_counts = {length: round(probabilities[i] * N) for i, length in enumerate(range(min_length, max_length + 1))}

    # Crear un diccionario de buckets por longitud
    length_buckets = {length: [] for length in range(min_length, max_length + 1)}
    for timestamp, trajectory in filtered_trajectories:
        length = len(trajectory)
        length_buckets[length].append((timestamp, trajectory))

    # Seleccionar trayectorias respetando las frecuencias deseadas
    selected_trajectories = []
    for length, target_count in target_counts.items():
        available = length_buckets[length]
        if len(available) >= target_count:
            selected_trajectories.extend(random.sample(available, target_count))
        else:
            selected_trajectories.extend(available)  # Agregar todas las disponibles si no son suficientes
            print(f"No hay suficientes trayectorias de l={length}, por favor usa una database más grande.")

    # Redistribuir las trayectorias faltantes
    while len(selected_trajectories) < N:
        remaining_lengths = [
            length for length, available in length_buckets.items()
            if len(available) > 0 and (length, available) not in selected_trajectories
        ]
        if not remaining_lengths:
            break
        for length in remaining_lengths:
            if len(selected_trajectories) >= N:
                break
            selected_trajectories.append(random.choice(length_buckets[length]))

    # Limitar el número de trayectorias seleccionadas a N
    selected_trajectories = selected_trajectories[:N]

    # Guardar las trayectorias seleccionadas en el archivo de salida con la fecha fija
    with open(output_file, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        for timestamp, trajectory in selected_trajectories:
            adjusted_timestamp = adjust_to_fixed_date(timestamp)
            writer.writerow([adjusted_timestamp, trajectory])

    # Graficar la distribución de longitudes seleccionadas
    lengths = [len(trajectory) for _, trajectory in selected_trajectories]
    plt.hist(lengths, bins=range(min_length, max_length + 2), align='left', rwidth=0.8, color='skyblue', edgecolor='black')
    plt.title('Length Distribution')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.xticks(range(min_length, max_length + 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Ejemplo de uso
def main():
    trajectory_file = "new_graph_mapped_trajectories.csv"
    min_length = 2
    max_length = 100
    N = 5000
    output_file = f"Porto_{max_length}_N{N}.csv"
    lambda_ = 0.5  # Parámetro de la distribución exponencial (ajustable)

    try:
        select_trajectories_exponential(trajectory_file, min_length, max_length, N, lambda_, output_file)
        print(f"Trayectorias seleccionadas guardadas en {output_file}")
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()






