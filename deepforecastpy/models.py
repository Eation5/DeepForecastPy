import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional, MultiHeadAttention, LayerNormalization, Embedding, Input, Concatenate

class BaseModel:
    """Base class for deep learning models, providing common functionalities like building, compiling, training, and predicting."""
    def __init__(self, input_shape, output_dim):
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.model = None

    def build_model(self, **kwargs):
        """Abstract method to be implemented by subclasses for model construction."""
        raise NotImplementedError("Subclasses must implement build_model method.")

    def compile_model(self, optimizer='adam', loss='mse', metrics=None):
        """Compiles the Keras model with specified optimizer, loss, and metrics."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train_model(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, callbacks=None):
        """Trains the model using provided training data and parameters."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
        print(f"\nTraining {self.__class__.__name__}...")
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                                 validation_split=validation_split, callbacks=callbacks, verbose=1)
        return history

    def predict(self, X_test):
        """Generates predictions for the given test data."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
        return self.model.predict(X_test)

    def summary(self):
        """Prints a summary of the model architecture."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
        self.model.summary()

class LSTMModel(BaseModel):
    """LSTM-based model for time series forecasting, supporting stacked and bidirectional layers."""
    def build_model(self, units=50, activation='relu', dropout=0.2, recurrent_dropout=0.2, num_layers=1, bidirectional=False):
        self.model = Sequential()
        for i in range(num_layers):
            return_sequences = True if i < num_layers - 1 else False
            if bidirectional:
                self.model.add(Bidirectional(LSTM(units, activation=activation, return_sequences=return_sequences,
                                                recurrent_dropout=recurrent_dropout), input_shape=self.input_shape if i == 0 else None))
            else:
                self.model.add(LSTM(units, activation=activation, return_sequences=return_sequences,
                                    recurrent_dropout=recurrent_dropout, input_shape=self.input_shape if i == 0 else None))
            self.model.add(Dropout(dropout))
        
        self.model.add(Dense(self.output_dim))
        print("\nLSTM Model Summary:")
        self.model.summary()

class GRUModel(BaseModel):
    """GRU-based model for time series forecasting, supporting stacked and bidirectional layers."""
    def build_model(self, units=50, activation='relu', dropout=0.2, recurrent_dropout=0.2, num_layers=1, bidirectional=False):
        self.model = Sequential()
        for i in range(num_layers):
            return_sequences = True if i < num_layers - 1 else False
            if bidirectional:
                self.model.add(Bidirectional(GRU(units, activation=activation, return_sequences=return_sequences,
                                               recurrent_dropout=recurrent_dropout), input_shape=self.input_shape if i == 0 else None))
            else:
                self.model.add(GRU(units, activation=activation, return_sequences=return_sequences,
                                   recurrent_dropout=recurrent_dropout, input_shape=self.input_shape if i == 0 else None))
            self.model.add(Dropout(dropout))
        
        self.model.add(Dense(self.output_dim))
        print("\nGRU Model Summary:")
        self.model.summary()

class TransformerBlock(tf.keras.layers.Layer):
    """A single Transformer block, combining multi-head self-attention and a feed-forward network."""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
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

class TransformerModel(BaseModel):
    """Transformer-based model for time series forecasting, handling sequential data with attention mechanisms."""
    def build_model(self, embed_dim=32, num_heads=2, ff_dim=32, num_transformer_blocks=2, dropout_rate=0.1):
        inputs = Input(shape=self.input_shape)
        # Project input features to embed_dim
        x = Dense(embed_dim)(inputs)
        
        for _ in range(num_transformer_blocks):
            x = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)(x)
        
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(ff_dim, activation="relu")(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(self.output_dim)(x)

        self.model = Model(inputs=inputs, outputs=outputs)
        print("\nTransformer Model Summary:")
        self.model.summary()

class AttentionModel(BaseModel):
    """Model incorporating attention mechanism for improved time series forecasting."""
    def build_model(self, lstm_units=64, dense_units=32, dropout_rate=0.2):
        inputs = Input(shape=self.input_shape)
        lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
        
        # Apply attention mechanism
        attention = Dense(1, activation='tanh')(lstm_out)
        attention = tf.keras.layers.Flatten()(attention)
        attention = tf.keras.layers.Activation('softmax')(attention)
        attention = tf.keras.layers.RepeatVector(lstm_units)(attention)
        attention = tf.keras.layers.Permute([2, 1])(attention)
        
        sent_representation = tf.keras.layers.Multiply()([lstm_out, attention])
        sent_representation = tf.keras.layers.Lambda(lambda xin: tf.keras.backend.sum(xin, axis=1))(sent_representation)
        
        x = Dense(dense_units, activation='relu')(sent_representation)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(self.output_dim)(x)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        print("\nAttention Model Summary:")
        self.model.summary()
