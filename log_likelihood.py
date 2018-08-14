import numpy as np
from scipy.optimize import minimize

ran = np.random.randint(0, 1000)
np.random.seed(ran)
# print(ran)
# np.random.seed(24)
num_product = 26
num_sample = 25
num_product_feature = 2

# create data
data_price = np.random.randint(0, 10, size = (num_sample, num_product))
data_ischeapest = np.random.rand(num_sample, num_product)
data_ischeapest = (data_ischeapest > 0.5).astype(int)

# poisson lambda
poission_lambda = np.random.randint(50, 100)
# parameter alpha
alpha = np.random.rand(num_product_feature + 1)
attractiveness_product_j = np.exp(alpha[0] - alpha[1] * data_price + alpha[2] * data_ischeapest)

# attractiveness_array of each product
attractiveness_product_no_purchase = 1
attractiveness_product_sample = np.sum(attractiveness_product_j, axis = 1)
p = np.empty_like(attractiveness_product_j)
new_poisson_lambda = np.empty_like(attractiveness_product_j)
for i in range(0, num_sample):
	p[i] = attractiveness_product_j[i] / (attractiveness_product_sample[i] + attractiveness_product_no_purchase)
	new_poisson_lambda[i] = poission_lambda * p[i]

# number of buying product_j
num_attracted_product_j = np.empty_like(attractiveness_product_j)
for i in range(0, num_sample):
	num_attracted_product_j[i] = np.array([np.random.poisson(new_poisson_lambda[i], num_product)])
# number of buying in each sample
num_attracted_product_sample = np.sum(num_attracted_product_j, axis = 1)


def log_likelihood_obj (v_ll):
	ll = 0
	poission_lambda_ll = v_ll[:1]
	alpha = v_ll[1:]
	v = np.ones([num_sample, num_product])
	for i in range(0, num_sample):
		for j in range(0, num_product):
			v[i][j] = np.exp(alpha[0] - alpha[1] * data_price[i][j] + alpha[2] * data_ischeapest[i][j])
	p_product = np.empty_like(v)
	for i in range(0, num_sample):
		p_product[i] = v[i] / (np.sum(v, axis = 1)[i] + attractiveness_product_no_purchase)
		for j in range(0, num_product):
			ll += num_attracted_product_j[i][j] * np.log(poission_lambda_ll * p_product[i][j])
			ll -= poission_lambda_ll * p_product[i][j]
			temp = int(num_attracted_product_j[i][j])
			chb_j_sum = 0
			for k in range(1, temp + 1):
				chb_j_sum += np.log(k)
			ll -= chb_j_sum
	return -ll


u_bound = (0, None)
bnds = ((0, None),)
for i in range(0, num_product_feature + 1):
	bnds = ((0, None),) + bnds
res = minimize(log_likelihood_obj, np.ones(num_product_feature + 1 + 1), method = 'SLSQP',
               bounds = bnds)

result_lambda = res.x[0:1]
result_alpha = res.x[1:]
# check result
attractiveness_product_j.shape = (num_sample * num_product)
attractiveness_caculate_j = np.exp(
	result_alpha[0] - result_alpha[1] * data_price + result_alpha[2] * data_ischeapest)
caculate_p = np.empty_like(attractiveness_caculate_j)
caculate_poisson_lambda = np.empty_like(attractiveness_caculate_j)
for i in range(0, num_sample):
	caculate_p[i] = attractiveness_caculate_j[i] / (
			np.sum(attractiveness_caculate_j, axis = 1)[i] + attractiveness_product_no_purchase)
	caculate_poisson_lambda[i] = result_lambda * caculate_p[i]

print('poission_lambda:\t{}'.format(poission_lambda))
print('alpha:\t{}'.format(alpha))
print('result_lambda:\t{}'.format(result_lambda))
print('result_alpha:\t{}'.format(result_alpha))
