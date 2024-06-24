import numpy as np

# Carregar o array de um arquivo de texto
loaded_arr = np.load('recorded_gameplay.npy')

# Fix do array para usar no algoritmo genético
if len(loaded_arr) >= 10000:
    fixedArray = loaded_arr[:10000]
    np.save('recorded_gameplay_fix.npy', fixedArray)

else:
    fixedArray = np.zeros(10000)
    fixedArray[:len(loaded_arr)] = loaded_arr
    np.save('recorded_gameplay_fix.npy', fixedArray)