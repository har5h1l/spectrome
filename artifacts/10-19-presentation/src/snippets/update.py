optimizer.zero_grad()
# Compute training loss
loss.backward()
# Check finite gradients
optimizer.step()
