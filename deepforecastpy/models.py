import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional, MultiHeadAttention, LayerNormalization, Embedding

class BaseModel:
    """Base class for deep learning models."""
    def __init__(self, input_shape, output_dim):
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.model = None

    def build_model(self, **kwargs):
        raise NotImplementedError("Subclasses must implement build_model method.")

    def compile_model(self, optimizer='adam', loss='mse', metrics=None):
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train_model(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, callbacks=None):
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                                 validation_split=validation_split, callbacks=callbacks, verbose=1)
        return history

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
        return self.model.predict(X_test)

class LSTMModel(BaseModel):
    """LSTM-based model for time series forecasting."""
    def build_model(self, units=50, activation='relu', dropout=0.2, recurrent_dropout=0.2, num_layers=1, bidirectional=False):
        self.model = Sequential()
        # Add multiple LSTM layers if specified
        for i in range(num_layers):
            return_sequences = True if i < num_layers - 1 else False
            if bidirectional:
                self.model.add(Bidirectional(LSTM(units, activation=activation, return_sequences=return_sequences,
                                                recurrent_dropout=recurrent_dropout), input_shape=self.input_shape))
            else:
                self.model.add(LSTM(units, activation=activation, return_sequences=return_sequences,
                                    recurrent_dropout=recurrent_dropout, input_shape=self.input_shape))
            self.model.add(Dropout(dropout))
        
        # Output layer
        self.model.add(Dense(self.output_dim))
        print("LSTM Model Summary:")
        self.model.summary()

class GRUModel(BaseModel):
    """GRU-based model for time series forecasting."""
    def build_model(self, units=50, activation='relu', dropout=0.2, recurrent_dropout=0.2, num_layers=1, bidirectional=False):
        self.model = Sequential()
        # Add multiple GRU layers if specified
        for i in range(num_layers):
            return_sequences = True if i < num_layers - 1 else False
            if bidirectional:
                self.model.add(Bidirectional(GRU(units, activation=activation, return_sequences=return_sequences,
                                               recurrent_dropout=recurrent_dropout), input_shape=self.input_shape))
            else:
                self.model.add(GRU(units, activation=activation, return_sequences=return_sequences,
                                   recurrent_dropout=recurrent_dropout, input_shape=self.input_shape))
            self.model.add(Dropout(dropout))
        
        # Output layer
        self.model.add(Dense(self.output_dim))
        print("GRU Model Summary:")
        self.model.summary()

class TransformerBlock(tf.keras.layers.Layer):
    """A single Transformer block for sequence processing."""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [
                Dense(ff_dim, activation="relu"),
                Dense(embed_dim),
            ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    """Embeds input tokens and adds positional encoding."""
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class TransformerModel(BaseModel):
    """Transformer-based model for time series forecasting."""
    def build_model(self, embed_dim=32, num_heads=2, ff_dim=32, num_transformer_blocks=2, maxlen=100, vocab_size=10000, dropout_rate=0.1):
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        # Assuming input_shape is (timesteps, features) and we want to embed features
        # For time series, we might treat each feature vector at a timestep as a 'token'
        # This is a simplified approach for demonstration.
        # A more complex approach would involve projecting features to embed_dim.
        
        # For simplicity, let's assume input_shape is (timesteps, 1) and we are embedding the single feature value
        # If input_shape is (timesteps, features), we need a different embedding strategy or flatten.
        # Let's adapt for a sequence of numerical features directly.
        
        # Project input features to embed_dim
        x = Dense(embed_dim)(inputs)
        
        # Add positional encoding (simplified, typically done with sine/cosine for numerical series)
        # For now, we'll just pass it through transformer blocks directly.
        
        for _ in range(num_transformer_blocks):
            x = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)(x)
        
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(ff_dim, activation="relu")(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(self.output_dim)(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        print("Transformer Model Summary:")
        self.model.summary()
