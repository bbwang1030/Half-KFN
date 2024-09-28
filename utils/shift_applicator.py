# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

from utils.data_utils import *

# -------------------------------------------------
# SHIFT APPLICATOR
# -------------------------------------------------


def apply_shift(X_te_orig, y_te_orig, shift, orig_dims, datset):
	X_te_1 = None
	y_te_1 = None
	
	if shift == 'rand':
		print('Randomized')
		X_te_1 = X_te_orig.copy()
		y_te_1 = y_te_orig.copy()
	elif shift == 'large_gn_shift_0.3':
		print('Large GN shift 0.3')
		normalization = 255.0
		X_te_1, _ = gaussian_noise_subset(X_te_orig, 100.0, normalization=normalization, delta_total=0.3)
		y_te_1 = y_te_orig.copy()
	elif shift == 'large_gn_shift_0.2':
		print('Large GN shift 0.2')
		normalization = 255.0
		X_te_1, _ = gaussian_noise_subset(X_te_orig, 100.0, normalization=normalization, delta_total=0.2)
		y_te_1 = y_te_orig.copy()
	elif shift == 'medium_gn_shift_0':
		print('Medium GN Shift 0')
		if datset != "simulation":
			normalization = 255.0
			X_te_1, _ = gaussian_noise_subset(X_te_orig, 1.0, normalization=normalization, delta_total=0)
			y_te_1 = y_te_orig.copy()
		else:
			normalization = 1.0
			X_te_1, _ = gaussian_noise_subset_simulation(X_te_orig, 2, normalization=normalization, delta_total=0)
			y_te_1 = y_te_orig.copy()
	elif shift == 'medium_gn_shift_0.01':
		print('Medium GN Shift 0.01')
		if datset != "simulation":
			normalization = 255.0
			X_te_1, _ = gaussian_noise_subset(X_te_orig, 1, normalization=normalization, delta_total=0.01)
			y_te_1 = y_te_orig.copy()
		else:
			normalization = 1.0
			X_te_1, _ = gaussian_noise_subset_simulation(X_te_orig, 10, normalization=normalization, delta_total=0.01)
			y_te_1 = y_te_orig.copy()
	elif shift == 'medium_gn_shift_0.05':
		print('Medium GN Shift 0.05')
		if datset != "simulation":
			normalization = 255.0
			X_te_1, _ = gaussian_noise_subset(X_te_orig, 1.0, normalization=normalization, delta_total=0.05)
			y_te_1 = y_te_orig.copy()
		else:
			normalization = 1.0
			X_te_1, _ = gaussian_noise_subset_simulation(X_te_orig, 10, normalization=normalization, delta_total=0.05)
			y_te_1 = y_te_orig.copy()
	elif shift == 'medium_gn_shift_0.1':
		print('Medium GN Shift 0.1')
		if datset != "simulation":
			normalization = 255.0
			X_te_1, _ = gaussian_noise_subset(X_te_orig, 1.0, normalization=normalization, delta_total=0.1)
			y_te_1 = y_te_orig.copy()
		else:
			normalization = 1
			X_te_1, _ = gaussian_noise_subset_simulation(X_te_orig, 10, normalization=normalization, delta_total=0.1)
			y_te_1 = y_te_orig.copy()
	elif shift == 'medium_gn_shift_0.2':
		print('Medium GN Shift 0.2')
		if datset != "simulation":
			normalization = 255.0
			X_te_1, _ = gaussian_noise_subset(X_te_orig, 1.0, normalization=normalization, delta_total=0.2)
			y_te_1 = y_te_orig.copy()
		else:
			normalization = 1.0
			X_te_1, _ = gaussian_noise_subset_simulation(X_te_orig, 1, normalization=normalization, delta_total=0.2)
			y_te_1 = y_te_orig.copy()
	elif shift == 'medium_gn_shift_0.3':
		print('Medium GN Shift 0.3')
		if datset != "simulation":
			normalization = 255.0
			X_te_1, _ = gaussian_noise_subset(X_te_orig, 1.0, normalization=normalization, delta_total=0.3)
			y_te_1 = y_te_orig.copy()
		else:
			normalization = 1.0
			X_te_1, _ = gaussian_noise_subset_simulation(X_te_orig, 1, normalization=normalization, delta_total=0.3)
			y_te_1 = y_te_orig.copy()
	elif shift == 'small_gn_shift_1.0':
		print('Small GN Shift 1')
		normalization = 255.0
		X_te_1, _ = gaussian_noise_subset(X_te_orig, 1.0, normalization=normalization, delta_total=0.4)
		y_te_1 = y_te_orig.copy()
	elif shift == 'large_gn_shift_0.1':
		print('Large GN shift 0.1')
		if datset != "simulation":
			normalization = 255.0
			X_te_1, _ = gaussian_noise_subset(X_te_orig, 100, normalization=normalization, delta_total=0.1)
			y_te_1 = y_te_orig.copy()
		else:
			normalization = 1.0
			X_te_1, _ = gaussian_noise_subset_simulation(X_te_orig, 20, normalization=normalization, delta_total=0.1)
			y_te_1 = y_te_orig.copy()
	elif shift == 'small_gn_shift_0.5':
		print('Small GN Shift 0.5')
		normalization = 255.0
		X_te_1, _ = gaussian_noise_subset(X_te_orig, 1.0, normalization=normalization, delta_total=0.2)
		y_te_1 = y_te_orig.copy()
	elif shift == 'large_gn_shift_0':
		print('Large GN shift 0')
		if datset != "simulation":
			normalization = 255.0
			X_te_1, _ = gaussian_noise_subset(X_te_orig, 100, normalization=normalization, delta_total=0)
			y_te_1 = y_te_orig.copy()
		else:
			normalization = 1.0
			X_te_1, _ = gaussian_noise_subset_simulation(X_te_orig, 20, normalization=normalization, delta_total=0)
			y_te_1 = y_te_orig.copy()
	elif shift == 'large_gn_shift_0.01':
		print('Large GN shift 0.01')
		if datset != "simulation":
			normalization = 255.0
			X_te_1, _ = gaussian_noise_subset(X_te_orig, 100, normalization=normalization, delta_total=0.01)
			y_te_1 = y_te_orig.copy()
		else:
			normalization = 1.0
			X_te_1, _ = gaussian_noise_subset_simulation(X_te_orig, 20, normalization=normalization, delta_total=0.01)
			y_te_1 = y_te_orig.copy()
	elif shift == 'large_gn_shift_0.05':
		print('Large GN shift 0.05')
		if datset != "simulation":
			normalization = 255.0
			X_te_1, _ = gaussian_noise_subset(X_te_orig, 100, normalization=normalization, delta_total=0.05)
			y_te_1 = y_te_orig.copy()
		else:
			normalization = 1.0
			X_te_1, _ = gaussian_noise_subset_simulation(X_te_orig, 20, normalization=normalization, delta_total=0.05)
			y_te_1 = y_te_orig.copy()
	elif shift == 'large_gn_shift_0.5':
		print('Large GN shift 0.5')
		if datset != "simulation":
			normalization = 255.0
			X_te_1, _ = gaussian_noise_subset(X_te_orig, 100, normalization=normalization, delta_total=0.5)
			y_te_1 = y_te_orig.copy()
		else:
			normalization = 1.0
			X_te_1, _ = gaussian_noise_subset_simulation(X_te_orig, 2, normalization=normalization, delta_total=0.5)
			y_te_1 = y_te_orig.copy()
	elif shift == 'small_gn_shift_0.1':
		print('Small GN Shift 0.1')
		normalization = 255.0
		X_te_1, _ = gaussian_noise_subset(X_te_orig, 1.0, normalization=normalization, delta_total=0)
		y_te_1 = y_te_orig.copy()
	elif shift == 'adversarial_shift_0':
		print('adversarial Shift 0 ')
		adv_samples, true_labels = adversarial_samples(datset)
		# print(adv_samples)
		X_te_1, y_te_1, _  = data_subset(X_te_orig, y_te_orig, adv_samples, true_labels, delta=0)
	elif shift == 'adversarial_shift_0.01':
		print('adversarial Shift 0.01')
		adv_samples, true_labels = adversarial_samples(datset)
		X_te_1, y_te_1, _ = data_subset(X_te_orig, y_te_orig, adv_samples, true_labels, delta=0.01)
	elif shift == 'adversarial_shift_0.05':
		print('adversarial Shift 0.05')
		adv_samples, true_labels = adversarial_samples(datset)
		X_te_1, y_te_1, _ = data_subset(X_te_orig, y_te_orig, adv_samples, true_labels, delta=0.05)
	elif shift == 'adversarial_shift_0.1':
		print('adversarial Shift 0.1')
		adv_samples, true_labels = adversarial_samples(datset)
		X_te_1, y_te_1, _ = data_subset(X_te_orig, y_te_orig, adv_samples, true_labels, delta=0.1)
	elif shift == 'adversarial_shift_0.2':
		print('adversarial Shift 0.2')
		adv_samples, true_labels = adversarial_samples(datset)
		X_te_1, y_te_1, _ = data_subset(X_te_orig, y_te_orig, adv_samples, true_labels, delta=0.2)
	elif shift == 'adversarial_shift_0.3':
		print(' adversarial Shift 0.3')
		adv_samples, true_labels = adversarial_samples(datset)
		X_te_1, y_te_1, _ = data_subset(X_te_orig, y_te_orig, adv_samples, true_labels, delta=0.3)
	elif shift == 'adversarial_shift_0.4':
		print(' adversarial Shift 0.4')
		adv_samples, true_labels = adversarial_samples(datset)
		X_te_1, y_te_1, _ = data_subset(X_te_orig, y_te_orig, adv_samples, true_labels, delta=0.4)
	elif shift == 'adversarial_shift_0.5':
		print(' adversarial Shift 0.5')
		adv_samples, true_labels = adversarial_samples(datset)
		X_te_1, y_te_1, _ = data_subset(X_te_orig, y_te_orig, adv_samples, true_labels, delta=0.5)
	elif shift == 'adversarial_shift_0.6':
		print(' adversarial Shift 0.6')
		adv_samples, true_labels = adversarial_samples(datset)
		X_te_1, y_te_1, _ = data_subset(X_te_orig, y_te_orig, adv_samples, true_labels, delta=0.6)
	elif shift == 'adversarial_shift_0.8':
		print(' adversarial Shift 0.8')
		adv_samples, true_labels = adversarial_samples(datset)
		X_te_1, y_te_1, _ = data_subset(X_te_orig, y_te_orig, adv_samples, true_labels, delta=0.8)
	elif shift == 'ko_shift_0.1':
		print('Small knockout shift')
		X_te_1, y_te_1 = knockout_shift(X_te_orig, y_te_orig, 0, 0)
	elif shift == 'ko_shift_0.5':
		print('Medium knockout shift')
		X_te_1, y_te_1 = knockout_shift(X_te_orig, y_te_orig, 0, 0.2)
	elif shift == 'ko_shift_1.0':
		print('Large knockout shift')
		X_te_1, y_te_1 = knockout_shift(X_te_orig, y_te_orig, 0, 0.4)
	elif shift == 'small_img_shift_0.1':
		print('Small image shift 0.1')
		X_te_1, _ = image_generator(X_te_orig, orig_dims, 10, 0.05, 0.05, 0.1, 0.1, False, False, delta=0)
		y_te_1 = y_te_orig.copy()
	elif shift == 'small_img_shift_0.5':
		print('Small image shift 0.5')
		X_te_1, _ = image_generator(X_te_orig, orig_dims, 10, 0.05, 0.05, 0.1, 0.1, False, False, delta=0.2)
		y_te_1 = y_te_orig.copy()
	elif shift == 'small_img_shift_1.0':
		print('Small image shift 1')
		X_te_1, _ = image_generator(X_te_orig, orig_dims, 10, 0.05, 0.05, 0.1, 0.1, False, False, delta=0.4)
		y_te_1 = y_te_orig.copy()
	elif shift == 'medium_img_shift_0.1':
		print('Medium image shift')
		X_te_1, _ = image_generator(X_te_orig, orig_dims, 40, 0.2, 0.2, 0.2, 0.2, True, False, delta=0)
		y_te_1 = y_te_orig.copy()
	elif shift == 'medium_img_shift_0.5':
		print('Medium image shift')
		X_te_1, _ = image_generator(X_te_orig, orig_dims, 40, 0.2, 0.2, 0.2, 0.2, True, False, delta=0.2)
		y_te_1 = y_te_orig.copy()
	elif shift == 'medium_img_shift_1.0':
		print('Medium image shift')
		X_te_1, _ = image_generator(X_te_orig, orig_dims, 40, 0.2, 0.2, 0.2, 0.2, True, False, delta=0.4)
		y_te_1 = y_te_orig.copy()
	elif shift == 'large_img_shift_0':
		print('Large image shift 0')
		X_te_1, _ = image_generator(X_te_orig, orig_dims, 90, 0.4, 0.4, 0.3, 0.4, True, True, delta=0)
		y_te_1 = y_te_orig.copy()
	elif shift == 'large_img_shift_0.01':
		print('Large image shift 0.01')
		X_te_1, _ = image_generator(X_te_orig, orig_dims, 90, 0.4, 0.4, 0.3, 0.4, True, True, delta=0.01)
		y_te_1 = y_te_orig.copy()
	elif shift == 'large_img_shift_0.05':
		print('Large image shift 0.05')
		X_te_1, _ = image_generator(X_te_orig, orig_dims, 90, 0.4, 0.4, 0.3, 0.4, True, True, delta=0.05)
		y_te_1 = y_te_orig.copy()
	elif shift == 'large_img_shift_0.1':
		print('Large image shift 0.1')
		X_te_1, _ = image_generator(X_te_orig, orig_dims, 90, 0.4, 0.4, 0.3, 0.4, True, True, delta=0.1)
		y_te_1 = y_te_orig.copy()
	elif shift == 'large_img_shift_0.2':
		print('Large image shift 0.2')
		X_te_1, _ = image_generator(X_te_orig, orig_dims, 90, 0.4, 0.4, 0.3, 0.4, True, True, delta=0.2)
		y_te_1 = y_te_orig.copy()
	elif shift == 'large_img_shift_0.3':
		print('Large image shift 0.3')
		X_te_1, _ = image_generator(X_te_orig, orig_dims, 90, 0.4, 0.4, 0.3, 0.4, True, True, delta=0.3)
		y_te_1 = y_te_orig.copy()
	elif shift == 'medium_img_shift_0.5+ko_shift_0.1':
		print('Medium image shift + knockout shift')
		X_te_1, _ = image_generator(X_te_orig, orig_dims, 40, 0.2, 0.2, 0.2, 0.2, True, False, delta=0.5)
		y_te_1 = y_te_orig.copy()
		X_te_1, y_te_1 = knockout_shift(X_te_1, y_te_1, 0, 0.1)
	elif shift == 'medium_img_shift_0.5+ko_shift_0.5':
		print('Medium image shift + knockout shift')
		X_te_1, _ = image_generator(X_te_orig, orig_dims, 40, 0.2, 0.2, 0.2, 0.2, True, False, delta=0.5)
		y_te_1 = y_te_orig.copy()
		X_te_1, y_te_1 = knockout_shift(X_te_1, y_te_1, 0, 0.5)
	elif shift == 'medium_img_shift_0.5+ko_shift_1.0':
		print('Medium image shift + knockout shift')
		X_te_1, _ = image_generator(X_te_orig, orig_dims, 40, 0.2, 0.2, 0.2, 0.2, True, False, delta=0.5)
		y_te_1 = y_te_orig.copy()
		X_te_1, y_te_1 = knockout_shift(X_te_1, y_te_1, 0, 1.0)
	elif shift == 'only_zero_shift+medium_img_shift_0.1':
		print('Only zero shift + Medium image shift')
		X_te_1, y_te_1 = only_one_shift(X_te_orig, y_te_orig, 0)
		X_te_1, _ = image_generator(X_te_1, orig_dims, 40, 0.2, 0.2, 0.2, 0.2, True, False, delta=0.1)
	elif shift == 'only_zero_shift+medium_img_shift_0.5':
		print('Only zero shift + Medium image shift')
		X_te_1, y_te_1 = only_one_shift(X_te_orig, y_te_orig, 0)
		X_te_1, _ = image_generator(X_te_1, orig_dims, 40, 0.2, 0.2, 0.2, 0.2, True, False, delta=0.5)
	elif shift == 'only_zero_shift+medium_img_shift_1.0':
		print('Only zero shift + Medium image shift')
		X_te_1, y_te_1 = only_one_shift(X_te_orig, y_te_orig, 0)
		X_te_1, _ = image_generator(X_te_1, orig_dims, 40, 0.2, 0.2, 0.2, 0.2, True, False, delta=1.0)
	
	return (X_te_1, y_te_1)