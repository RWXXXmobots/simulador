from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os
import glob
# from simulador import simulador
from eletrico_DJI import simulador
import numpy as np

class CaseAlreadyProcessed(Exception):
    """Exception raised when a case has already been processed."""
    def __init__(self, message="This case has already been processed."):
        self.message = message
        super().__init__(self.message)


def worker_task(target_mask: str, area: float, speed: float, width: float):

    input_file = f'rotas-condicionadas/{target_mask}/mission.csv'

    if not os.path.exists(input_file):
        raise FileNotFoundError
        return

    output_folder = lambda drone: f'resultados-eletrico/{drone}/vel{int(speed*10):02}/{target_mask}/'

    print(target_mask)
    
    #   Drone T20P - Bateria T20P          ------------------------------------------------------------------
    folder = output_folder('T20P_T20P')
    os.makedirs(folder, exist_ok=True)        
    if not glob.glob(os.path.join(folder, '*.xlsx')):
        num_motor = 4
        volume_tanque = np.arange(20,20.1,1)        #[L]
        paralelos = np.arange(1,1.1,1)
        M_bat = 6
        cap_max = 13*0.7
        E_bat_max = cap_max*3.7
        serie = 1
        K = 1.477559
        A =  K/num_motor
        simulador(input_file, folder, 'mission.csv', area, speed, width, num_motor, volume_tanque, paralelos, M_bat, cap_max, E_bat_max, serie, A)
        
    #   Drone T20P - Bateria T50           ------------------------------------------------------------------
    folder = output_folder('T20P_T50')
    os.makedirs(folder, exist_ok=True)        
    if not glob.glob(os.path.join(folder, '*.xlsx')):
        num_motor = 4
        volume_tanque = np.arange(20,20.1,1)        #[L]
        paralelos = np.arange(1,1.1,1)
        M_bat = 12.1
        cap_max = 30*0.7
        E_bat_max = cap_max*3.7*14
        serie = 1
        K = 1.477559
        A =  K/num_motor
        simulador(input_file, folder, 'mission.csv', area, speed, width, num_motor, volume_tanque, paralelos, M_bat, cap_max, E_bat_max, serie, A)
    
    #   Drone T25 - Bateria T25        ------------------------------------------------------------------
    folder = output_folder('T25_T25')
    os.makedirs(folder, exist_ok=True)        
    if not glob.glob(os.path.join(folder, '*.xlsx')):
        num_motor = 4
        volume_tanque = np.arange(20,20.1,1)        #[L]
        paralelos = np.arange(1,1.1,1)
        M_bat = 6.6
        cap_max = 15*0.7
        E_bat_max = cap_max*3.7*14
        serie = 1
        K = 1.26676
        A =  K/num_motor
        simulador(input_file, folder, 'mission.csv', area, speed, width, num_motor, volume_tanque, paralelos, M_bat, cap_max, E_bat_max, serie, A)
    
    #   Drone T25 - Bateria T50        ------------------------------------------------------------------
    folder = output_folder('T25_T50')
    os.makedirs(folder, exist_ok=True)        
    if not glob.glob(os.path.join(folder, '*.xlsx')):
        num_motor = 4
        volume_tanque = np.arange(20,20.1,1)        #[L]
        paralelos = np.arange(1,1.1,1)
        M_bat = 12.1
        cap_max = 30*0.7
        E_bat_max = cap_max*3.7*14
        serie = 1
        K = 1.26676
        A =  K/num_motor
        simulador(input_file, folder, 'mission.csv', area, speed, width, num_motor, volume_tanque, paralelos, M_bat, cap_max, E_bat_max, serie, A)
    
    #   Drone T40 - Bateria T40        ------------------------------------------------------------------
    folder = output_folder('T40_T40')
    os.makedirs(folder, exist_ok=True)        
    if not glob.glob(os.path.join(folder, '*.xlsx')):
        num_motor = 8
        volume_tanque = np.arange(40,40.1,1)        #[L]
        paralelos = np.arange(1,1.1,1)
        M_bat = 12.1
        cap_max = 30*0.7
        E_bat_max = cap_max*3.7*14
        serie = 1
        K = 1.477559
        A =  K/num_motor
        simulador(input_file, folder, 'mission.csv', area, speed, width, num_motor, volume_tanque, paralelos, M_bat, cap_max, E_bat_max, serie, A)
    
    #   Drone T40 - Bateria T40 x 2        ------------------------------------------------------------------
    folder = output_folder('T40_2T40')
    os.makedirs(folder, exist_ok=True)        
    if not glob.glob(os.path.join(folder, '*.xlsx')):
        num_motor = 8
        volume_tanque = np.arange(40,40.1,1)        #[L]
        paralelos = np.arange(1,1.1,1)
        M_bat = 12.1*2
        cap_max = 30*2*0.7
        E_bat_max = cap_max*3.7*14
        serie = 1
        K = 1.477559
        A =  K/num_motor
        simulador(input_file, folder, 'mission.csv', area, speed, width, num_motor, volume_tanque, paralelos, M_bat, cap_max, E_bat_max, serie, A)
    
    #   Drone T50 - Bateria T50        ------------------------------------------------------------------
    folder = output_folder('T50_T50')
    os.makedirs(folder, exist_ok=True)        
    if not glob.glob(os.path.join(folder, '*.xlsx')):
        num_motor = 8
        volume_tanque = np.arange(50,50.1,1)        #[L]
        paralelos = np.arange(1,1.1,1)
        M_bat = 12.1
        cap_max = 30*0.7
        E_bat_max = cap_max*3.7*14
        serie = 1
        K = 1.477559
        A =  K/num_motor
        simulador(input_file, folder, 'mission.csv', area, speed, width, num_motor, volume_tanque, paralelos, M_bat, cap_max, E_bat_max, serie, A)
    
    #   Drone T50 - Bateria T50 x 2        ------------------------------------------------------------------
    folder = output_folder('T50_2T50')
    os.makedirs(folder, exist_ok=True)        
    if not glob.glob(os.path.join(folder, '*.xlsx')):
        num_motor = 8
        volume_tanque = np.arange(50,50.1,1)        #[L]
        paralelos = np.arange(1,1.1,1)
        M_bat = 12.1*2
        cap_max = 30*2*0.7
        E_bat_max = cap_max*3.7*14
        serie = 1
        K = 1.477559
        A =  K/num_motor
        simulador(input_file, folder, 'mission.csv', area, speed, width, num_motor, volume_tanque, paralelos, M_bat, cap_max, E_bat_max, serie, A)

if __name__ == '__main__':

    subareas = [16, 36, 64, 100]
    percentages = [2, 6, 10, 14, 18]
    width = [0, 6, 12]
    indexes = [0]
    modes = ['TSP', 'LM']
    speeds = [0.5, 1.0, 2.0, 4.0]

    target_mask = lambda sba, p, w, i, m: f'field_{sba}ha_100ha_{p}%_{w}m_{i}_{m}'

    tasks = [
        (target_mask(sba, p, w, i, m), sba, speed, w) 
            for sba in subareas 
            for p in percentages 
            for w in width 
            for i in indexes 
            for m in modes 
            for speed in speeds
    ]

    #worker_task(*tasks[0])
    #exit()

    with ProcessPoolExecutor(max_workers=4) as executor:

        futures = [executor.submit(worker_task, *mask) for mask in tasks]

        for future in as_completed(futures):
            try:
                future.result()
            except FileNotFoundError:
                print("Case Not Found")
            except CaseAlreadyProcessed:
                print("Case Already Processed. Skiping")
            except Exception as e:
                print(e)


