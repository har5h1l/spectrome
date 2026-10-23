K = response["K"]
model_out = (response["eigenvectors"][..., :, 1:K]
             * frequency_response[..., None, 1:K]).sum(dim=-1)
