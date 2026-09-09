_base_ = ['./masa_detic_swinb_open_vocabulary_test_dual.py']

model = dict(tracker=dict(assignment_mode='hungarian_legacy'))
