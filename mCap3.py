import numpy as np
import scipy
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Embedding, Dense, Dropout
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
from tensorflow.keras.layers import ReLU, Concatenate, Conv1D, RepeatVector, Embedding, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout, Reshape,Softmax, AveragePooling2D
from tensorflow.keras.layers import MultiHeadAttention, TimeDistributed
from tensorflow.keras.layers import LayerNormalization

from tensorflow.keras.models import load_model, Model

from tensorflow.keras import regularizers
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from tensorflow.keras.layers import LSTM as RNN

from tensorflow.keras.utils import to_categorical

initialvar = -8
small = 1e-7


class Patches(layers.Layer):
    def __init__(self, patch_size=16, name='name'):
        super(Patches, self).__init__(name=name)
        self.patch_size = patch_size
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches



class NLL(layers.Layer):
    def __init__(self, numclass=2,usemask=False, name='name'):
        super(NLL, self).__init__(name=name)
        self.numclass = numclass
    def call(self, x):
        evidence, y = x[0], x[1]
        y1hot = tf.one_hot(y, self.numclass)
        opinion = evidence+1.
        S = tf.reduce_sum(opinion, axis=-1, keepdims=True)
        ep = tf.divide(opinion, S) 
        
        term1 = tf.square(tf.cast(y1hot, dtype='float32') - ep)
        term2 = ep*(1.-ep) / (S+1.)
        loss = tf.reduce_sum(term1+term2, axis=-1, keepdims=True)
        return loss
class NLL2(layers.Layer):
    def __init__(self, numclass=2,usemask=False, name='name'):
        super(NLL2, self).__init__(name=name)
        self.numclass = numclass
    def call(self, x):
        evidence, y = x[0], x[1]
        y1hot = tf.one_hot(y, self.numclass)
        opinion = evidence+1.
        S = tf.reduce_sum(opinion, axis=-1, keepdims=True)
        term1 = y1hot * (tf.math.log(S) - tf.math.log(opinion))
        loss = tf.reduce_sum(term1, axis=-1, keepdims=True)
        return loss

class KL(layers.Layer):
    def __init__(self, numclass=2,usemask=False, name='name'):
        super(KL, self).__init__(name=name)
        self.numclass = numclass
        self.lgammak = scipy.special.loggamma(numclass)
    def call(self, x):
        evidence, y = x[0], x[1]
        y1hot = tf.one_hot(y, self.numclass)
        opinion = evidence+1.
        tilde_a = y1hot + (1.-y1hot)*opinion
        sum_a = tf.reduce_sum(tilde_a, axis=-1, keepdims=True)
        
        term1 = tf.math.lgamma(sum_a) 
        term2 = tf.reduce_sum(tf.math.lgamma(tilde_a), axis=-1, keepdims=True)
        term3 = tf.math.digamma(tilde_a) - tf.math.digamma(sum_a)
        
        term4 = tf.reduce_sum((tilde_a-1)*term3, axis=-1, keepdims=True)
        output = tf.reduce_sum(term1 - term2, axis=-1, keepdims=True) + term4 - self.lgammak
        return output


def ViTevidential(datas, dims, bnparam):
    embed_dim = dims[0]
    ff_dim = dims[1]
    nlabel = dims[2]
    trn1 = bnparam[1][0]
    trn2 = bnparam[1][1]
    
    f4 = AveragePooling2D(pool_size=(2,2),padding="valid", name='f4')(datas[0])
    f5 = Patches(patch_size=8,name='f5')(f4)
    f5c = ImgEmbedding(T=196, embed_dim=embed_dim, name='f5c')(f5)
    """
    f5 = Patches(patch_size=14,name='f5')(datas[0])
    f5c = ImgEmbedding(T=16*16, embed_dim=embed_dim, name='f5c')(f5)
    """
    for i in range(2):
        attn1 = MultiHeadAttention(num_heads=2,key_dim=embed_dim, trainable=trn2,name='attn'+str(i))(query=f5c, value=f5c, key=f5c)
        add1a = Lambda(lambda x: x[0]+x[1], name='adda'+str(i)) ([f5c, attn1])
        ln1 = LayerNormalization(trainable=trn2,name='lna'+str(i)) (add1a)
        fc1 = Dense(embed_dim, activation="linear", name='ff'+str(i))(ln1)
        add1b = Lambda(lambda x: x[0]+x[1], name='addb'+str(i)) ([ln1, fc1])
        f5c = LayerNormalization(trainable=trn2,name='lnb'+str(i)) (add1b)
    key = Lambda(lambda x: tf.reduce_mean(x, axis=-2), name='f5d') (f5c)
    for i in range(1):
        fc = Dense(embed_dim, activation="relu", name='fc'+str(i))(key)
        fd = Dense(embed_dim, activation="linear", name='fd'+str(i))(fc)
        key = LayerNormalization(trainable=trn2,name='out'+str(i)) (fd)
    lnow0 = Dense(2*nlabel, activation="exponential", name='lnow0')(key)
    lnow = Lambda(lambda x: tf.reshape(x, [-1,nlabel,2]), name='lnow') (lnow0)
    """
    return lnow
    """
    nll = NLL2(numclass=2, name='nll')([lnow, datas[1]])
    kl = KL(name='kl')([lnow, datas[1]])
    return [lnow, nll, kl]

def ViTevi(datas, dims, bnparam, cnn):
    embed_dim = dims[0]
    ff_dim = dims[1]
    nlabel = dims[2]
    trn1 = bnparam[1][0]
    trn2 = bnparam[1][1]
    f4 = cnn(datas[0])
    i=0
    f5 = Conv2D(embed_dim,3,activation='linear',padding="same",name='fa'+str(i),
                kernel_regularizer=L2(1e-7),bias_regularizer=L2(1e-7),)(f4)
    """
    f5 = Conv2D(embed_dim,3,activation='relu',padding="same",name='fb'+str(i),
                kernel_regularizer=L2(1e-7),bias_regularizer=L2(1e-7),)(f5)
    """
    f4 = MaxPooling2D(pool_size=(2,2),padding="valid", name='fe'+str(i))(f5)
    key = Flatten(name='f5c')(f4)
    for i in range(1):
        fc = Dense(embed_dim, activation="relu", name='fc'+str(i),
                   kernel_regularizer=L2(1e-7),bias_regularizer=L2(1e-7),)(key)
        fd = Dense(embed_dim, activation="linear", name='fd'+str(i),
                   kernel_regularizer=L2(1e-7),bias_regularizer=L2(1e-7),)(fc)
        fd = LayerNormalization(name='out'+str(i)) (fd)
    lnow0 = Dense(nlabel*2, activation="exponential", name='lnow0',
                 kernel_regularizer=L2(1e-7),bias_regularizer=L2(1e-7),)(fd)
    lnow = Lambda(lambda x: tf.reshape(x, [-1,nlabel,2]), name='lnow') (lnow0)
    nll = NLL2(numclass=2, name='nll')([lnow, datas[1]])
    kl = KL(name='kl')([lnow, datas[1]])
    return [lnow, nll, kl]


def ViTvgg(datas, dims, bnparam, cnn):
    embed_dim = dims[0]
    ff_dim = dims[1]
    nlabel = dims[2]
    trn1 = bnparam[1][0]
    trn2 = bnparam[1][1]
    f4 = cnn(datas[0])
    
    i=0
    f5 = Conv2D(embed_dim,3,activation='linear',padding="same",name='fa'+str(i),
                kernel_regularizer=L2(1e-7),bias_regularizer=L2(1e-7),)(f4)
    """
    f5 = Conv2D(embed_dim,3,activation='relu',padding="same",name='fb'+str(i),
                kernel_regularizer=L2(1e-7),bias_regularizer=L2(1e-7),)(f5)
    """
    f4 = MaxPooling2D(pool_size=(2,2),padding="valid", name='fe'+str(i))(f5)
    
    key = Flatten(name='f5c')(f4)
    for i in range(1):
        fc = Dense(embed_dim, activation="relu", name='fc'+str(i),
                   kernel_regularizer=L2(1e-7),bias_regularizer=L2(1e-7),)(key)
        fd = Dense(embed_dim, activation="relu", name='fd'+str(i),
                   kernel_regularizer=L2(1e-7),bias_regularizer=L2(1e-7),)(fc)
    lnow = Dense(nlabel, activation="sigmoid", name='lnow',
                 kernel_regularizer=L2(1e-7),bias_regularizer=L2(1e-7),)(fd)
    return lnow



def ViTcnn(datas, dims, bnparam):
    embed_dim = dims[0]
    ff_dim = dims[1]
    nlabel = dims[2]
    trn1 = bnparam[1][0]
    trn2 = bnparam[1][1]
    f5 = Conv2D(64,3,activation='relu',padding="same",name='fa',
                kernel_regularizer=L2(1e-7),bias_regularizer=L2(1e-7),)(datas[0])
    f4 = MaxPooling2D(pool_size=(2,2),padding="valid", name='fe')(f5)
    for i in range(4): 
        f5 = Conv2D(ff_dim[i],3,activation='relu',padding="same",name='fa'+str(i),
                    kernel_regularizer=L2(1e-7),bias_regularizer=L2(1e-7),)(f4)
        f5 = Conv2D(ff_dim[i],3,activation='relu',padding="same",name='fb'+str(i),
                    kernel_regularizer=L2(1e-7),bias_regularizer=L2(1e-7),)(f5)
        f4 = MaxPooling2D(pool_size=(2,2),padding="valid", name='fe'+str(i))(f5)
    key = Flatten(name='f5c')(f4)

    for i in range(1):
        fc = Dense(embed_dim, activation="relu", name='fc'+str(i),
                   kernel_regularizer=L2(1e-8),bias_regularizer=L2(1e-8),)(key)
        fc = LayerNormalization(name='out'+str(i)) (fc)
        fd = Dense(embed_dim, activation="relu", name='fd'+str(i),
                   kernel_regularizer=L2(1e-8),bias_regularizer=L2(1e-8),)(fc)
    
    lnow = Dense(nlabel, activation="sigmoid", name='lnow',
                 kernel_regularizer=L2(1e-8),bias_regularizer=L2(1e-8),)(fd)
    return lnow

class SeqEmbedding(Layer):
    def __init__(self, T, V, embed_dim, **kwargs):
        super(SeqEmbedding, self).__init__(**kwargs)
        self.token_embeddings = Embedding(
            input_dim=V, output_dim=embed_dim,
            embeddings_regularizer=L2(1e-7),)
        self.position_embeddings = Embedding(
            input_dim=T, output_dim=embed_dim,
            embeddings_regularizer=L2(1e-7),)
        self.T = T
        self.V = V
        self.embed_dim = embed_dim
    def call(self, inputs):
        positions = tf.range(start=0, limit=self.T, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        output = embedded_tokens + embedded_positions
        return output
    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

class ImgEmbedding(Layer):
    def __init__(self, T, embed_dim, rescale=1., **kwargs):
        super(ImgEmbedding, self).__init__(**kwargs)
        self.position_embeddings = Embedding(
            input_dim=T, output_dim=embed_dim,
            embeddings_regularizer=L2(1e-7),)
        self.T = T
        self.rescale = rescale
        self.embed_dim = embed_dim
    def call(self, inputs):
        positions = tf.range(start=0, limit=self.T, delta=1)
        embedded_positions = self.position_embeddings(positions)
        output = inputs * self.rescale + embedded_positions
        return output

class KeyEmbedding(Layer):
    def __init__(self, T, embed_dim, **kwargs):
        super(KeyEmbedding, self).__init__(**kwargs)
        self.position_embeddings = Embedding(
            input_dim=T, output_dim=embed_dim,
            embeddings_regularizer=L2(1e-7),)
        self.T = T
        self.embed_dim = embed_dim
    def call(self, inputs):
        positions = tf.range(start=0, limit=self.T, delta=1)
        embedded_positions = self.position_embeddings(positions)
        keymask = tf.expand_dims(tf.cast(inputs, dtype='float32'), axis=-1)
        output = keymask * embedded_positions
        return output
    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

class Decoder(Layer):
    def __init__(self, embed_dim, num_heads, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention1 = MultiHeadAttention(num_heads, key_dim=embed_dim,
                                             kernel_regularizer=L2(1e-7),)
        self.attention2 = MultiHeadAttention(num_heads, key_dim=embed_dim,
                                             kernel_regularizer=L2(1e-7),)
        self.dense1 = Dense(embed_dim, activation="relu",
                            kernel_regularizer=L2(1e-7),)
        self.dense2 = Dense(embed_dim, activation=None,
                            kernel_regularizer=L2(1e-7),)
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.layernorm3 = LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_mask(inputs)
        """
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)
        """
        if mask is not None:
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)
        
        attn1 = self.attention1(
            query=inputs, value=inputs, key=inputs,
            attention_mask=combined_mask)
        out1 = self.layernorm1(inputs + attn1)
        
        attn2 = self.attention2(
            query=out1, value=encoder_outputs, key=encoder_outputs,
            attention_mask=padding_mask,) 
        out2 = self.layernorm2(out1 + attn2)

        ff_output = self.dense1(out2)
        ff_output = self.dense2(ff_output)
        output = self.layernorm3(out2 + ff_output)
        
        return output

    def get_causal_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, T = input_shape[0], input_shape[1]
        i = tf.range(T)[:, tf.newaxis]
        j = tf.range(T)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, T, T))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype="int32")],
            axis=0,)
        output = tf.tile(mask, mult)
        return output
class DecoderKey(Layer):
    def __init__(self, embed_dim, num_heads, **kwargs):
        super(DecoderKey, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention2 = MultiHeadAttention(num_heads, key_dim=embed_dim,
                                             kernel_regularizer=L2(1e-7),)
        self.dense1 = Dense(embed_dim, activation="relu",
                            kernel_regularizer=L2(1e-7),)
        self.dense2 = Dense(embed_dim, activation=None,
                            kernel_regularizer=L2(1e-7),)
        self.layernorm2 = LayerNormalization()
        self.layernorm3 = LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, keymask, mask=None):
        """
        causal_mask = self.get_causal_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)
        """
        if mask is not None:
            padding_mask = tf.cast(mask[:,:, tf.newaxis], dtype="int32")
            keymask = tf.cast(keymask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, keymask)
        
        attn2 = self.attention2(
            query=inputs, value=encoder_outputs, key=encoder_outputs,
            attention_mask=padding_mask,) 
        out2 = self.layernorm2(inputs + attn2)

        ff_output = self.dense1(out2)
        ff_output = self.dense2(ff_output)
        output = self.layernorm3(out2 + ff_output)
        return output

def ViTcaptionVgg(datas, dims, bnparam, cnn):
    embed_dim = dims[0]
    ff_dim = dims[1]
    keysize = dims[2]
    vocsize = dims[3]
    Tcaption = dims[4]
    trn1 = bnparam[1][0]
    trn2 = bnparam[1][1]
    f4 = cnn(datas[0])
    f5c = Reshape((100,embed_dim), name='reshape')(f4)
    f5c = ImgEmbedding(T=100, embed_dim=embed_dim, rescale=1., name='f5c')(f5c)
    
    seq = SeqEmbedding(T=Tcaption, V=vocsize, embed_dim=embed_dim, name='embedcap')(datas[1])
    encoderkey = KeyEmbedding(T=keysize, embed_dim=embed_dim, name='encoderkey')(datas[2])
    cross = Lambda(lambda x: tf.concat(x, axis=-2), name='cross') ([f5c, encoderkey])
    for i in range(3):
        seq = Decoder(embed_dim, num_heads=2, name='decoderl'+str(i))(seq, cross)
    
    for i in range(1):
        fd = TimeDistributed(Dense(embed_dim, activation="relu",
                                   kernel_regularizer=L2(1e-7),), name='finalc'+str(i))(seq)
    lnow = TimeDistributed(Dense(vocsize, activation="softmax",
                                 kernel_regularizer=L2(1e-7),), name='lnow')(fd)
    return lnow
def ViTcaptionVgg2(datas, dims, bnparam, cnn):
    embed_dim = dims[0]
    ff_dim = dims[1]
    keysize = dims[2]
    vocsize = dims[3]
    Tcaption = dims[4]
    trn1 = bnparam[1][0]
    trn2 = bnparam[1][1]
    f4 = cnn(datas[0])
    f5c = Reshape((100,ff_dim), name='reshape')(f4)
    f5c = ImgEmbedding(T=100, embed_dim=ff_dim, rescale=1., name='f5c')(f5c)
    
    seq = SeqEmbedding(T=Tcaption, V=vocsize, embed_dim=embed_dim, name='embedcap')(datas[1])
    key = KeyEmbedding(T=keysize, embed_dim=embed_dim, name='encoderkey')(datas[2])
    keymask = Lambda(lambda x: x>0, name='keymask')(datas[2])
    for i in range(2):
        seq = Decoder(embed_dim, num_heads=2, name='decoderl'+str(i))(seq, f5c)
        seq = DecoderKey(embed_dim, num_heads=2, name='decoderk'+str(i))(seq, key, keymask)
    
    for i in range(1):
        fd = TimeDistributed(Dense(embed_dim, activation="relu",
                                   kernel_regularizer=L2(1e-7),), name='finalc'+str(i))(seq)
    lnow = TimeDistributed(Dense(vocsize, activation="softmax",
                                 kernel_regularizer=L2(1e-7),), name='lnow')(fd)
    return lnow

