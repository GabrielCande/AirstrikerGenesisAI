import numpy as np

# Carregar o array de um arquivo de texto
loaded_arr = np.load('recorded_gameplay.npy')

print("Array carregado:")
print(loaded_arr)
converte_txt = np.savetxt('NPYparaTXT.txt', loaded_arr, fmt='%f')