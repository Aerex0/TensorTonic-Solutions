def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    output=[]
    for row in X:
        row_output=[]
        for neuron in range(len(W[0])):
            sum =0
            for j in range(len(row)):
                sum += row[j]*W[j][neuron]
            sum += b[neuron]
            row_output.append(sum)
        output.append(row_output)
    return output