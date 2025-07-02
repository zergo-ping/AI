import torch

# 1.1 Создание тензоров
#----------------------------------------------------------------
# - Тензор размером 3x4, заполненный случайными числами от 0 до 1
tensor_3x4 = torch.rand(3,4)
print("\nТензор размером 3x4, заполненный случайными числами от 0 до 1\n", tensor_3x4)


# - Тензор размером 2x3x4, заполненный нулями
tensor_2x3x4 = torch.zeros(2, 3, 4 )
print("\nТензор размером 2x3x4, заполненный нулями\n", tensor_2x3x4)


# - Тензор размером 5x5, заполненный единицами
tensor_5x5 = torch.ones(5, 5)
print("\nТензор размером 5x5, заполненный единицами\n", tensor_5x5)


# - Тензор размером 4x4 с числами от 0 до 15 (используйте reshape)
tensor_4x4 = torch.arange(16).reshape(4, 4)
print("\nТензор размером 4x4 с числами от 0 до 15\n", tensor_4x4)





# 1.2 Операции с тензорами
#----------------------------------------------------------------
# Дано: тензор A размером 3x4 и тензор B размером 4x3
tensor_A = torch.rand(3, 4)
tensor_B = torch.rand(4, 3)

# - Транспонирование тензора A
A_transposed = torch.transpose(tensor_A, 0, 1) 
print("Транспонированный A:\n", A_transposed)

# - Матричное умножение A и B
matrix_product = torch.matmul(tensor_A, tensor_B)  
print("\nМатричное произведение A и B:\n", matrix_product)

# - Поэлементное умножение A и транспонированного B
B_transposed = torch.transpose(tensor_B, 0, 1)
# Расширяем A до (3,4,1) и B_transposed до (1,3,4)
elementwise_product = tensor_A * B_transposed
print("\nПоэлементное произведение A и B.T:\n", elementwise_product)

# - Вычислите сумму всех элементов тензора A
sum_A = torch.sum(tensor_A)
print("\nСумма всех элементов A:", sum_A.item())




# 1.3 Индексация и срезы
#----------------------------------------------------------------
# Создайте тензор размером 5x5x5
tensor_5x5x5 = torch.rand(5, 5, 5)

# - Первую строку
first_str = tensor_5x5x5[0, :, :]
print("\nПервая строка\n" , first_str)

# - Последний столбец
last_column = tensor_5x5x5[:, :, -1]
print("\nПоследний столбец\n" , last_column)

# - Подматрицу размером 2x2 из центра тензора
center_submatrix = tensor_5x5x5[2:4, 2:4, 2]
print("\nПодматрицу размером 2x2 из центра тензора\n" , center_submatrix)

# - Все элементы с четными индексами
even_elements = tensor_5x5x5[::2, ::2, ::2]
print("\nВсе элементы с четными индексами\n" , even_elements)



# 1.4 Работа с формами 
#----------------------------------------------------------------
# Создайте тензор размером 24 элемента
tensor_Size24 = torch.rand(24)

# - 2x12
print("\nФорма 2x12\n" , tensor_Size24.reshape(2, 12))

# - 3x8
print("\nФорма 3x8\n" , tensor_Size24.reshape(3, 8))

# - 4x6
print("\nФорма 4x6\n" , tensor_Size24.reshape(4, 6))


# - 2x3x4
print("\nФорма 2x3x4\n" , tensor_Size24.reshape(2, 3, 4))

# - 2x2x2x3
print("\nФорма 2x2x2x3\n" , tensor_Size24.reshape(2, 2, 2, 3))


