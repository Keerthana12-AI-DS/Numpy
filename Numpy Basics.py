#NumPy is one of the most fundamental libraries in Python for scientific computing

#Create Numpy Arrays Using Lists or Tuples

import numpy as np

my_list = [1, 2, 3, 4, 5]
numpy_array = np.array(my_list)
print("Simple NumPy Array:",numpy_array)


#Initialize a Python NumPy Array Using Special Functions

import numpy as np

zeros_array = np.zeros((2, 3))
ones_array = np.ones((3, 3))
constant_array = np.full((2, 2), 7)
range_array = np.arange(0, 10, 2)  # start, stop, step
linspace_array = np.linspace(0, 1, 5)  # start, stop, num

print("Zero Array:","\n",zeros_array)
print("Ones Array:","\n",ones_array)
print("Constant Array:","\n",constant_array)
print("Range Array:","\n",range_array)
print("Linspace Array:","\n",linspace_array)


#Create Python Numpy Arrays Using Random Number Generation

import numpy as np

random_array = np.random.rand(2, 3)
normal_array = np.random.randn(2, 2)
randint_array = np.random.randint(1, 10, size=(2, 3))  

print(random_array)
print(normal_array)
print(randint_array)

#Create Python Numpy Arrays Using Matrix Creation Routines

import numpy as np

identity_matrix = np.eye(3)
diagonal_array = np.diag([1, 2, 3])
zeros_like_array = np.zeros_like(diagonal_array)
ones_like_array = np.ones_like(diagonal_array)

print(identity_matrix)
print(diagonal_array)
print(zeros_like_array)
print(ones_like_array)


#Array indexing in NumPy
#Accessing Elements in 1D Arrays

import numpy as np

arr = np.array([10, 20, 30, 40, 50])
print(arr[0])


# Accessing Elements in Multidimensional Arrays

import numpy as np 

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix[1, 2])


#slicing
#array

import numpy as np

arr = np.array([0, 1, 2, 3, 4, 5])
print(arr[1:4])

#multidimensional

import numpy as np

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix[0:2, 1:3])



#binary operation

import numpy as np

array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([6, 7, 8, 9, 10])

add_result = np.add(array1, array2)
multiply_result = array1 * array2
print(f"Addition: {add_result}")
print(f"Multiplication: {multiply_result}")

matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])
matrix_product = np.matmul(matrix1, matrix2)
print(f"Matrix Product:\n{matrix_product}")

sqrt_array = np.sqrt(array1)
print(f"Square Root: {sqrt_array}")


#Aggregation Functions

import numpy as np

data = np.array([10, 20, 30, 40, 50, 60, 70])

average_value = np.mean(data)
highest_value = np.max(data)
lowest_value = np.min(data)
data_range = np.ptp(data)
spread_of_data = np.std(data)

print("--- NumPy Array Operations ---")
print(f"Original Array: {data}")
print(f"Mean (Average): {average_value:.2f}")
print(f"Maximum Value: {highest_value:.2f}")
print(f"Minimum Value: {lowest_value:.2f}")
print(f"Range (Max - Min): {data_range:.2f}")
print(f"Standard Deviation: {spread_of_data:.2f}")
