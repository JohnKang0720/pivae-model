import tensorflow as tf
from keras.layers import Lambda
import numpy as np
from matplotlib import pyplot as plt

N_DIM = 100
N_SAMPLES = 3000

fig,ax = plt.subplots(1,3, figsize=(15,5))

def slice_func(x, start, size):
    """Utility function. We use it to take a slice of tensor start from 'start' with length 'size'. Search tf.slice for detailed use of this function.
    
    """
    return tf.slice(x, [0,start],[-1,size])

def perm_func(x, ind):
    """Utility function. Permute x with given indices. Search tf.gather for detailed use of this function.
    """
    return tf.gather(x, indices=ind, axis=-1)

def realnvp_layer(x_input):
    DD = x_input.shape[-1]; ## DD needs to be an even number
    dd = (DD//2);
    
    ## define some lambda functions
    clamp_func = Lambda(lambda x: 0.1*tf.tanh(x));
    trans_func = Lambda(lambda x: x[0]*tf.exp(x[1]) + x[2]);
    sum_func = Lambda(lambda x: tf.reduce_sum(-x, axis=-1, keepdims=True));
    
    ## compute output for s and t functions
    x_input1 = Lambda(slice_func, arguments={'start':0,'size':dd})(x_input);
    x_input2 = Lambda(slice_func, arguments={'start':dd,'size':dd})(x_input);
    st_output = x_input1;
    
    n_nodes = [dd//2, dd//2, DD];
    act_func = ['relu', 'relu', 'linear'];
    for ii in range(len(act_func)):
        st_output = tf.keras.layers.Dense(n_nodes[ii], activation = act_func[ii])(st_output);
    s_output = Lambda(slice_func, arguments={'start':0,'size':dd})(st_output);
    t_output = Lambda(slice_func, arguments={'start':dd,'size':dd})(st_output);
    s_output = clamp_func(s_output); ## keep small values of s
    
    ## perform transformation
    trans_x = trans_func([x_input2, s_output, t_output]);
    output = tf.keras.layers.concatenate([trans_x, x_input1], axis=-1);
    return output

def realnvp_block(x_output):
    for _ in range(2):
        x_output = realnvp_layer(x_output);
    return x_output

def simulate_cont_data(length, n_dim):
    ## simulate 2d z
    np.random.seed(777);
    
    u_true = np.random.uniform(0,2*np.pi,size = [length,1]);
    mu_true = np.hstack((u_true, 2*np.sin(u_true)));
    z_true = np.random.normal(0, 0.6, size=[length,2])+mu_true;
    z_true = np.hstack((z_true, np.zeros((z_true.shape[0],n_dim-2))));
    
    ## simulate mean
    dim_x = z_true.shape[-1];
    permute_ind = [];
    n_blk = 4;
    for ii in range(n_blk):
        np.random.seed(ii);
        permute_ind.append(tf.convert_to_tensor(np.random.permutation(dim_x)));
    
    x_input = tf.keras.layers.Input(shape=(dim_x,));
    x_output = realnvp_block(x_input);
    for ii in range(n_blk-1):
        x_output = Lambda(perm_func, arguments={'ind':permute_ind[ii]})(x_output);
        x_output = realnvp_block(x_output);
    
    realnvp_model = tf.keras.Model(inputs=[x_input], outputs=x_output);
    mean_true = realnvp_model.predict(z_true)
    lam_true = np.exp(2.2*np.tanh(mean_true));
    return z_true, u_true, mean_true, lam_true


z_true, u_true, _, lam_true = simulate_cont_data(N_SAMPLES,N_DIM)
x_true = tf.random.poisson(shape=(), lam=lam_true)

print(x_true.shape)
ax[0].scatter(x=z_true[:, 0], y=z_true[:, 1], c=u_true)
# plt.show()

## -- MODEL -- 

# GIN - the projectoin from m dimension to n Poisson spikes

class FirstLayerGIN(tf.keras.layers.Layer):
    def __init__(self, observed_dim, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.observed_dim = observed_dim
        self.n_nodes = [max(30, observed_dim // 4), max(30, observed_dim // 4), observed_dim - latent_dim]   # observed_dim - z_dim (2)
        self.activations = ["relu", "relu", "linear"]

        # define Dense layers once
        self.denses = [
            tf.keras.layers.Dense(self.n_nodes[i], activation=self.activations[i])
            for i in range(len(self.n_nodes))
        ]

    def call(self, z):
        j = z
        for dense in self.denses:
            j = dense(j)
        output = tf.keras.layers.concatenate([z, j], axis=-1)
        return output
    
class GINLayer(tf.keras.layers.Layer):
    def __init__(self, n_blocks, observed_dim, **kwargs):
        super().__init__(**kwargs)
        self.n_blocks = n_blocks
        self.n_nodes = [30, 30, 30]   # last size depends on input
        self.activations = ["relu", "relu", "linear"]
        self.observed_dim = observed_dim
        # Dense blocks except the last one (we’ll set last in __init__ when we know input size)
        self.denses = [
            tf.keras.layers.Dense(self.n_nodes[i], activation=self.activations[i])
            for i in range(n_blocks)
        ]

        # placeholder, will be set dynamically in build()
        self.final_dense = None

        self.sum_func = tf.keras.layers.Lambda(lambda x: -1 * tf.reduce_sum(x, axis=-1, keepdims=True))
        self.trans_func = tf.keras.layers.Lambda(lambda x: x[0] * tf.exp(x[1]) + x[2])
 
    def build(self, input_shape): # builds the last layer 
        N = int(input_shape[-1])
        n = N // 2
        final_units = 2 * (N - n) - 1
        self.final_dense = tf.keras.layers.Dense(final_units, activation="softplus")
        super().build(input_shape)

    # copied from reference
    def slice_func(self, x, start, size):
        # return tf.slice(x, [0, start], [-1, size])
        if size is None:
            return x[..., start:]
        return x[..., start:start + size]

    # copied from reference
    def perm_func(self,x, ind):
        """Utility function. Permute x with given indices. Search tf.gather for detailed use of this function.
        """
        return tf.gather(x, indices=ind, axis=-1)
    
    def call(self, x):
        N = x.shape[-1]
        n = N // 2

        x_input1 = self.slice_func(x, 0, n)
        x_input2 = self.slice_func(x, n, N - n)

        st = x_input1

        for dense in self.denses:
            st = dense(st)
        st = self.final_dense(st)

        s_output = self.slice_func(st, 0, N - n - 1)
        t_output = self.slice_func(st, N - n - 1, N - n)

        s_output = 0.1 * tf.tanh(s_output)  # clamp
        s_output = tf.keras.layers.concatenate([s_output, self.sum_func(s_output)], axis=-1)

        transformed = self.trans_func([x_input2, s_output, t_output])
        return tf.keras.layers.concatenate([transformed, x_input1], axis=-1)


class Encoder(tf.keras.Model):
    def __init__(self, observed_dim, latent_dim):
        super(Encoder, self).__init__()
        self.observed_dim = observed_dim
        self.latent_dim = latent_dim

        # define layers once
        self.d1 = tf.keras.layers.Dense(60, activation='tanh')
        self.d2 = tf.keras.layers.Dense(60, activation='tanh')
        # self.d3 = tf.keras.layers.Dense(60, activation='tanh')


        # mean / log-variance heads: linear outputs (no ReLU)
        self.z_mean_layer = tf.keras.layers.Dense(self.latent_dim, activation=None)
        self.z_log_var_layer = tf.keras.layers.Dense(self.latent_dim, activation=None)

    def call(self, x):
        x = tf.cast(x, tf.float32)
        x = self.d1(x)
        x = self.d2(x)
        # x = self.d3(x)
        z_mean = self.z_mean_layer(x)
        z_log_var = self.z_log_var_layer(x)
        return z_mean, z_log_var


class Prior(tf.keras.Model):
    def __init__(self, latent_dim):
        super(Prior, self).__init__()
        self.latent_dim = latent_dim

        self.d1 = tf.keras.layers.Dense(20, activation='tanh')
        self.d2 = tf.keras.layers.Dense(20, activation='tanh')
        # self.d3 = tf.keras.layers.Dense(32, activation='tanh')
        self.lam_mean_layer = tf.keras.layers.Dense(self.latent_dim, activation=None)
        self.lam_log_var_layer = tf.keras.layers.Dense(self.latent_dim, activation=None)

    def call(self, u):
        u = tf.cast(u, tf.float32)
        u = self.d1(u)
        u = self.d2(u)
        # u = self.d3(u)
        lam_mean = self.lam_mean_layer(u)
        lam_log_var = self.lam_log_var_layer(u)
        return lam_mean, lam_log_var


class Decoder(tf.keras.Model):
    def __init__(self, observed_dim, latent_dim):
        super(Decoder, self).__init__()
        self.observed_dim = observed_dim
        self.latent_dim = latent_dim
        self.first_layer_gin = FirstLayerGIN(observed_dim=observed_dim, latent_dim=latent_dim)
        self.gin_layer = GINLayer(3, observed_dim=observed_dim)

        # self.out = tf.keras.layers.Dense(self.observed_dim, activation="softplus")

    def call(self, z):
        z = tf.cast(z, tf.float32)
        # x = self.d1(x)
        # x = self.d2(x)
        z = self.first_layer_gin(z)
        for _ in range(2):
            z = self.gin_layer(z)
        # output = self.out(z)
        return z

class PIVAE(tf.keras.Model):
    def __init__(self, observed_dim, latent_dim, kl_weight=1.0):
        super(PIVAE, self).__init__()
        self.observed_dim = observed_dim
        self.latent_dim = latent_dim
        self.encoder = Encoder(observed_dim=observed_dim, latent_dim=latent_dim)
        self.decoder = Decoder(observed_dim=observed_dim, latent_dim=latent_dim)
        self.prior = Prior(latent_dim=latent_dim)
        self.kl_weight = kl_weight

    def compute_posterior_params(self, z_mean, z_log_var, lam_mean, lam_log_var):
        # numerically stable combination of two Gaussians (posterior merging)
        diff_log_var = z_log_var - lam_log_var
        post_mean = (z_mean / (1.0 + tf.exp(diff_log_var))) + (lam_mean / (1.0 + tf.exp(-diff_log_var)))
        post_log_var = z_log_var + lam_log_var - tf.math.log(tf.clip_by_value(tf.exp(z_log_var) + tf.exp(lam_log_var), 1e-7, 1e7))
        return post_mean, post_log_var


    def reparameterize(self, mean, log_var):
        eps = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * eps

    def call(self, x, u):
        u_input = tf.expand_dims(u, axis=-1)
        lam_mean, lam_log_var = self.prior(u_input)
        z_mean, z_log_var = self.encoder(x)
        post_mean, post_log_var = self.compute_posterior_params(z_mean, z_log_var, lam_mean, lam_log_var)
        s = self.reparameterize(post_mean, post_log_var)
        x_pred = self.decoder(s)
        # return prediction and posterior params for loss computation
        return x_pred, post_mean, post_log_var, lam_mean, lam_log_var

    def compute_kl_loss(self, post_mean, post_log_var, lam_mean, lam_log_var):
        kl_per_dim = 1 + post_log_var - lam_log_var - ((tf.square(post_mean - lam_mean) + tf.exp(post_log_var)) / tf.exp(lam_log_var))
        kl = -0.5 * tf.reduce_sum(kl_per_dim, axis=-1)  # sum over dims, shape (batch,)
        return kl              

    def train_step(self, data):
        x_input, y_target = data

        # x_input = tf.convert_to_tensor(x_input)
        # y_target = tf.convert_to_tensor(y_target)
        x_input = tf.cast(x_input, tf.float32)
        y_target = tf.cast(y_target, tf.float32)

        with tf.GradientTape() as tape:
            # tape.watch(x_input)
            x_pred, post_mean, post_log_var, lam_mean, lam_log_var = self(x_input, y_target, training=True)
            
            # reconstruction loss: sum squared error per sample, then mean batch
            recon_per_sample = tf.reduce_sum(x_pred - x_input*tf.math.log(tf.clip_by_value(x_pred, 1e-7, 1e7)), axis=-1)
   
            # KL loss
            kl_loss = self.compute_kl_loss(post_mean, post_log_var, lam_mean, lam_log_var)

            total_loss = recon_per_sample + self.kl_weight * kl_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
    
        return {"loss": tf.reduce_mean(total_loss), "recon_loss": tf.reduce_mean(recon_per_sample), "kl_loss": tf.reduce_mean(kl_loss)}


# Example usage (fill N_SAMPLES, N_DIM, x_true, u_true appropriately)
pivae = PIVAE(N_DIM, 2)
pivae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4))
fit = pivae.fit(x_true, u_true, epochs=50, batch_size=64, verbose=2)


z_mean, z_log_var = pivae.encoder(x_true)
latent_samples = pivae.reparameterize(z_mean, z_log_var).numpy()


ax[1].scatter(x=latent_samples[:, 0], y=latent_samples[:, 1], c=u_true)
history = fit.history['loss']
ax[2].plot(history[:])
plt.show()