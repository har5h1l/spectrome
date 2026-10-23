_, output = nt.evaluate_spectral_response_torch(response, Fe_graph)
return output.T[subject["observed_regions"].to(device)]
