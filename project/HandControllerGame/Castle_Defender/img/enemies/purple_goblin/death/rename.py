import os

for n in range(20):
    file_num = '00' + str(n)
    file_num = file_num[-3:]
    os.rename(f'5_enemies_1_die_{file_num}.png', f'{n}.png')