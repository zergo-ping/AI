import torch

#2.1 Простые вычисления с градиентами
#-------------------------------------------------------
# Создаем тензоры с requires_grad=True
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = torch.tensor(4.0, requires_grad=True)

# Вычисляем функцию f(x,y,z) = x² + y² + z² + 2xyz
f = x**2 + y**2 + z**2 + 2*x*y*z

# Вычисляем градиенты
f.backward()

# Получаем градиенты
grad_x = x.grad
grad_y = y.grad
grad_z = z.grad

print(f"Градиенты: df/dx = {grad_x.item()}, df/dy = {grad_y.item()}, df/dz = {grad_z.item()}")

# Аналитическая проверка
def analytical_gradients(x_val, y_val, z_val):
    df_dx = 2*x_val + 2*y_val*z_val
    df_dy = 2*y_val + 2*x_val*z_val
    df_dz = 2*z_val + 2*x_val*y_val
    return df_dx, df_dy, df_dz

# Конвертируем тензоры в числа для проверки
x_val = x.item()
y_val = y.item()
z_val = z.item()

analytical_dx, analytical_dy, analytical_dz = analytical_gradients(x_val, y_val, z_val)

print(f"Аналитические градиенты: df/dx = {analytical_dx}, df/dy = {analytical_dy}, df/dz = {analytical_dz}")

# Проверка 
assert torch.allclose(torch.tensor([grad_x, grad_y, grad_z]), 
                     torch.tensor([analytical_dx, analytical_dy, analytical_dz]))
print("Градиенты совпадают\n")

# 2.2 Градиент функции потерь
#---------------------------------------------------------
# Реализация MSE с автоматическим вычислением градиентов
def mse_loss(x, y_true, w, b):
    """
    x: входные данные (тензор)
    y_true: истинные значения (тензор)
    w: вес (тензор с requires_grad=True)
    b: смещение (тензор с requires_grad=True)
    """
    # Предсказание: y_pred = w*x + b
    y_pred = w * x + b
    
    # Вычисление MSE: (1/n) * Σ(y_pred - y_true)^2
    loss = torch.mean((y_pred - y_true)**2)
    return loss

# Пример данных
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y_true = torch.tensor([2.0, 4.0, 6.0, 8.0])  # Идеальные значения для w=2, b=0

# Параметры модели (инициализируем с requires_grad=True)
w = torch.tensor(1.0, requires_grad=True)  # начальное значение веса
b = torch.tensor(0.0, requires_grad=True)  # начальное значение смещения

# Вычисляем потери
loss = mse_loss(x, y_true, w, b)

# Вычисляем градиенты
loss.backward()

# Получаем градиенты
grad_w = w.grad
grad_b = b.grad

print(f"MSE loss: {loss.item():.4f}")
print(f"Градиент по w: {grad_w.item():.4f}")
print(f"Градиент по b: {grad_b.item():.4f}")

# Аналитическая проверка градиентов
def analytical_gradients(x, y_true, w_val, b_val):
    n = len(x)
    y_pred = w_val * x + b_val
    dw = (2/n) * torch.sum((y_pred - y_true) * x)
    db = (2/n) * torch.sum(y_pred - y_true)
    return dw, db

# Конвертируем в обычные числа для проверки
w_val = w.item()
b_val = b.item()

analytical_dw, analytical_db = analytical_gradients(x, y_true, w_val, b_val)

print("\nАналитические градиенты:")
print(f"∂MSE/∂w: {analytical_dw:.4f}")
print(f"∂MSE/∂b: {analytical_db:.4f}")

# Проверка совпадения
print("\nПроверка совпадения\n")
assert torch.allclose(grad_w, analytical_dw, atol=1e-4)
assert torch.allclose(grad_b, analytical_db, atol=1e-4)


#2.3 Цепное правило
#---------------------------------------
def composite_function(x):
    """Составная функция f(x) = sin(x² + 1)"""
    return torch.sin(x**2 + 1)

# Создаем точку для вычисления градиента
x = torch.tensor(2.0, requires_grad=True)

# Вычисление градиента вторым способом (autograd.grad)
f2 = composite_function(x)
grad_autograd = torch.autograd.grad(outputs=f2, inputs=x, retain_graph=True)[0].clone()

# Аналитическое решение
def analytical_gradient(x_tensor):
    """Аналитический градиент df/dx = 2x * cos(x² + 1)"""
    return 2 * x_tensor * torch.cos(x_tensor**2 + 1)

# Вычисляем аналитический градиент
analytical_grad = analytical_gradient(x).clone()

# Вывод результатов
print(f"Точка x: {x.item()}")
print("\nРезультаты вычисления градиента:")
print(f"Метод autograd.grad: {grad_autograd.item():.4f}")
print(f"Аналитический результат: {analytical_grad.item():.4f}")

# Проверка совпадения
def allclose(a, b):
    return torch.allclose(a, b, atol=1e-4)
