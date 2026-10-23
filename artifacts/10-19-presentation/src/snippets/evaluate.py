model.eval()
with torch.no_grad():
    filters = model(omega)
    analytical = predict_spectra(held_out)
    learned = predict_spectra(held_out, filters)
    analytical_score = analytical_scores[held_out_index]
    learned_score = subject_correlation(held_out, filters).item()
