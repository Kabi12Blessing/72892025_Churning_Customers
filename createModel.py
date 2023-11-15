# Define your MLP model with the best hyperparameters

def create_mlp_model(input_dim, hidden_layer_sizes=(64,), activation='relu'):
    input_layer = Input(shape=(input_dim,))
    x = input_layer
    for layer_size in hidden_layer_sizes:
        x = Dense(layer_size, activation=activation)(x)
    output_layer = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
