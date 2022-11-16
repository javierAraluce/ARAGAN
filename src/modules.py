import tensorflow as tf

def summary_tensorboard(summary_writer,
                        dataset_split, 
                        tensorboard_step,
                        gen_loss,
                        disc_loss,
                        total_loss_gan,
                        kld_metric,
                        mae_metric,
                        mse_metric,     
                        correlation_coefficient, 
                        auc, 
                        leraning_rate):  
        
        with summary_writer.as_default():
            tf.summary.scalar(dataset_split + '_gen_loss', 
                              gen_loss, 
                              step = tensorboard_step)
            tf.summary.scalar(dataset_split + '_disc_loss', 
                              disc_loss, 
                              step = tensorboard_step)
            tf.summary.scalar(dataset_split + '_total_loss_gan', 
                              total_loss_gan, 
                              step = tensorboard_step)
            tf.summary.scalar(dataset_split + '_lr', 
                              leraning_rate, 
                              step = tensorboard_step)                             
            tf.summary.scalar(dataset_split + '_kld_metric',
                              kld_metric,
                              step = tensorboard_step)
            tf.summary.scalar(dataset_split + '_mae_metric',
                              mae_metric, 
                              step = tensorboard_step)
            tf.summary.scalar(dataset_split + '_mse_metric', 
                              mse_metric, 
                              step = tensorboard_step)  
            tf.summary.scalar(dataset_split + '_correlation_coefficient', 
                              correlation_coefficient, 
                              step = tensorboard_step)
            tf.summary.scalar(dataset_split + '_auc', 
                              auc, 
                              step = tensorboard_step)  
            
def pearson_r(y_true : tf.Tensor,
              y_pred : tf.Tensor,):
        '''_summary_

        Args:
            y_true (tf.Tensor): _description_
            y_pred (tf.Tensor): _description_

        Returns:
            _type_: _description_
        '''
        
        x = y_true
        y = y_pred
        mx = tf.reduce_mean(x, axis=1, keepdims=True)
        my = tf.reduce_mean(y, axis=1, keepdims=True)
        xm, ym = x - mx, y - my
        t1_norm = tf.nn.l2_normalize(xm, axis = 1)
        t2_norm = tf.nn.l2_normalize(ym, axis = 1)
        cosine = tf.compat.v1.losses.cosine_distance(t1_norm, t2_norm, axis = 1)
        return cosine