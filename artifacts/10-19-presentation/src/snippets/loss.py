filters = model(omega)
train_scores = torch.stack([
    subject_correlation(subject, filters)
    for subject in training_subjects
])
loss = 1 - train_scores.mean()
