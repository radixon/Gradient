import torch

print('------------------------------------------')
print('-----------------f(x) = c-----------------')
print('----------------f\'(x) = 0----------------')
print('----------Constant Function Rule----------')
"""
    Constant Function
        Given f(x) = c, the derivative of f(x) is give by
            f'(x) = 0

    Parameters:
        x:  The function argument
        c:  A constant value that is defaulted to zero

    Note:
        x is multiplied by 0.  Torch find a derivative without a function argument
"""

x = torch.tensor(1.0, requires_grad=True)
c = torch.tensor(3.0)
y = x*0 + c
y.backward()
print(f'f(x) = 3.0')
print(f'f\'(1.0) = {x.grad} \n')


c = torch.pi
y = x*0 + c
y.backward()
print(f'f(x) = π')
print(f'f\'(1.0) = {x.grad} \n')


print('------------------------------------------')
print('-----------------f(x) = x-----------------')
print('----------------f\'(x) = 1----------------')
print('----------Identity Function Rule----------')
"""
    Identity Function: 
        Given f(x) = x, the derivative of f(x) is given by
            f'(x) = 1
        
        Parameters:
            x:  The function argument 
            c:  A constant value that is defaulted to zero
"""


x = torch.tensor(1000.0, requires_grad=True)
c = torch.tensor(0.0)
y = x + c
y.backward()
print(f'f(x) = x')
print(f'f\'(1000) = {x.grad} \n')

x = torch.tensor(25.0, requires_grad=True)
c = torch.tensor(3.0)
y = x + c
y.backward()
print(f'f(x) = x + 3.0')
print(f'f\'(25.0) = {x.grad} \n')


print('------------------------------------------')
print('----------------f(x) = c * x----------------')
print('----------------f\'(x) = c----------------')
print('----------Constant Multiple Rule----------')
"""
    Constant Multiple Rule: 
        Given f(x) = c*x, the derivative of f(x) is given by
            f'(x) = c
        
        Parameters:
            x:  The function argument
            c:  A constant value that is defaulted to zero
"""


x = torch.tensor(25.0, requires_grad=True)
c = torch.tensor(3.0)
y = c * x
y.backward()
print(f'f(x) = 3.0 * x')
print(f'f\'(25.0) = {x.grad} \n')

x = torch.tensor(1000.0, requires_grad=True)
c = torch.pi
y = c * x
y.backward()
print(f'f(x) = π * x')
print(f'f\'(1000.0) = {x.grad} \n')


print('------------------------------------------')
print('---------------f(x) =  x**n---------------')
print('----------f\'(x) = n * x**(n - 1)----------')
print('----------------Power Rule----------------')
"""
    Power Rule: 
        Given f(x) = x**n, the derivative of f(x) is given by
            f'(x) = n * x**(n - 1)
        
        Parameters:
            x: function argument
"""


x = torch.tensor(4.0, requires_grad=True)
y = x**3
y.backward()
print(f'f(x) = x**3')
print(f'f\'(4.0) = {x.grad} \n')

x = torch.tensor(7.0, requires_grad=True)
y = 2 * x**4
y.backward()
print(f'f(x) = 2 * x**4')
print(f'f\'(7.0) = {x.grad} \n')


print('------------------------------------------')
print('---------(f + g)(x) = f(x) + g(x)---------')
print('------(f + g)\'(x) = f\'(x) + g\'(x)------')
print('-----------------Sum Rule-----------------')
"""
    Sum Rule: 
        Given (f + g)(x), the derivative of (f + g)(x) is given by
            (f + g)'(x) = f'(x) + g'(x)
        
        Parameters:
            x: function argument
"""


x = torch.tensor(35.0, requires_grad=True)
y = 5*x**2 + 7 * x - 6
y.backward()
print(f'f(x) = 5x**2 + 7x - 6')
print(f'f\'(35.0) = {x.grad} \n')

x = torch.tensor(8.0, requires_grad=True)
y = x**3 + 6*x**2 + 8*x + 1
y.backward()
print(f'f(x) = x**3 + 6x**2 + 8x + 1')
print(f'f\'(8.0) = {x.grad} \n')


print('------------------------------------------')
print('---------(f - g)(x) = f(x) - g(x)---------')
print('------(f - g)\'(x) = f\'(x) - g\'(x)------')
print('--------------Difference Rule--------------')
"""
    Difference Rule: 
        Given (f - g)(x), the derivative of (f - g)(x) is given by
            (f - g)'(x) = f'(x) - g'(x)
        
        Parameters:
            x: function argument
"""


x = torch.tensor(2.0, requires_grad=True)
y = 4*x**6 - 3*x**5 - 10*x**2 + 5*x + 16
y.backward()
print(f'f(x) = 4x**6 - 3x**5 - 10x**2 + 5x + 16')
print(f'f\'(2.0) = {x.grad} \n')

x = torch.tensor(3.0, requires_grad=True)
y = x**3 + 6*x**2 + 8*x + 1
y.backward()
print(f'f(x) = x**3 - 6x**2 - 8x - 1')
print(f'f\'(3.0) = {x.grad} \n')


print('------------------------------------------')
print('---------(f * g)(x) = f(x) * g(x)---------')
print('(f * g)\'(x) = f(x) * g\'(x) + f\'(x) * g(x)')
print('---------------Product Rule---------------')
"""
Difference Rule: 
    Given (f * g)(x), the derivative of (f * g)(x) is given by
        (f * g)'(x) = f(x) * g'(x) + f'(x) * g(x)
    
    Parameters:
        x: function argument
"""


x = torch.tensor(5.0, requires_grad=True)
y = (x**4 - 1)*(x**2 + 1)
y.backward()
print(f'f(x) = (x**4 - 1)(x**2 + 1)')
print(f'f\'(5.0) = {x.grad} \n')

x = torch.tensor(10.0, requires_grad=True)
y = (x**2 + 17)*(x**3 - 3*x + 1)
y.backward()
print(f'f(x) = (x**2 + 17)(x**3 - 3*x + 1)')
print(f'f\'(10.0) = {x.grad} \n')
