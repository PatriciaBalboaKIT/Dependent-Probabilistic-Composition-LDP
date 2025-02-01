import csv
import pandas as pd
import numpy as np
from ast import literal_eval

# Función para cargar los timestamps desde el archivo
def load_timestamps(csv_file):
   # Cargar trayectorias originales
   trajectories = pd.read_csv(csv_file, header=None).values.tolist()
   timestamps = []     
   for trajectory in trajectories:
        init_time = int(trajectory[0])
        coordinates = literal_eval(trajectory[1])
        for i in range(len(coordinates)):
            timestamps.append(init_time + 15 * i)
   return timestamps
   

        

# Función para calcular los intervalos de tiempo de una hora
def calculate_time_intervals(timestamps, output_csv):
    print("total timestamps",len(timestamps))
    min_timestamp = min(timestamps)
    max_timestamp = max(timestamps)
    
    print(f"Timestamp mínimo: {min_timestamp}")
    print(f"Timestamp máximo: {max_timestamp}")
    
    # Crear intervalos de una hora (3600s)
    # Crear los intervalos en Unix timestamps
    intervals = np.arange(min_timestamp, max_timestamp+3600 , 3600)
    print(intervals)
    
    # Crear DataFrame con intervalos de tiempo
    # Crear el DataFrame directamente desde los intervalos Unix
    time_intervals_df = pd.DataFrame({
        'start': intervals[:-1],  # Todos los valores excepto el último
        'end': intervals[1:]      # Todos los valores excepto el primero
        })
    
    # Guardar los intervalos en un archivo CSV
    time_intervals_df.to_csv(output_csv, index=False)
    print(f"Intervalos de tiempo guardados en '{output_csv}'")

    return time_intervals_df

# Main
if __name__ == '__main__':
    # Archivo de entrada con el formato descrito
    input_file = 'Dresden_databases/Dresden_5_N5000.csv'  # Cambia esto por la ruta a tu archivo
    output_file = 'Dresden_time_intervals_5.csv'
    
    # Cargar los timestamps
    timestamps = load_timestamps(input_file)
    
    # Calcular intervalos de tiempo y guardarlos en un CSV
    time_intervals_df = calculate_time_intervals(timestamps, output_file)




