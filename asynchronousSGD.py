import tensorflow as tf

tf.app.flags.DEFINE_integer("task_index", 0, "Index of the worker task")
FLAGS = tf.app.flags.FLAGS

g = tf.Graph()

with g.as_default():
    with tf.device("/job:worker/task:0"):
        weight = tf.Variable(tf.ones([33762578]), name="model")

    # For every VM use the respective tfrecords and calculate the sparse gradient
    with tf.device("/job:worker/task:%d" % FLAGS.task_index):
        filepaths = [["00", "01", "02", "03", "04"], ["05", "06", "07", "08", "09"], ["10", "11", "12", "13", "14"],
                     ["15", "16", "17", "18", "19"], ["20", "21"]]
        filename_queue = tf.train.string_input_producer(
            ["/home/ubuntu/tfrecords/tfrecords" + path for path in filepaths[FLAGS.task_index]],
            num_epochs=None)

        reader = tf.TFRecordReader()
        _, data_serialized = reader.read(filename_queue, name='reader_%d' % FLAGS.task_index)

        features = tf.parse_single_example(data_serialized,
                                           features={
                                               'label': tf.FixedLenFeature([1], dtype=tf.int64),
                                               'index': tf.VarLenFeature(dtype=tf.int64),
                                               'value': tf.VarLenFeature(dtype=tf.float32),
                                           })

        gradient_sparse = tf.SparseTensor(shape=[33762578],
                                          indices=[features['index'].values],
                                          values=tf.mul(tf.mul(tf.mul(
                                              tf.sigmoid(
                                                  tf.mul(tf.cast(features['label'], tf.float32), tf.reduce_sum(
                                                      tf.mul(tf.gather(weight, features['index'].values),
                                                             features['value'].values)))) - 1,
                                              features['value'].values),
                                              tf.cast(features['label'], tf.float32)), -0.01))

    # Collect the gradients tp form the model and calculate the error on the test sample
    with tf.device("/job:worker/task:0"):
        assign_op = weight.assign_add(
            tf.sparse_to_dense(tf.transpose(gradient_sparse.indices), [33762578], gradient_sparse.values))

        filename_queue_two = tf.train.string_input_producer(
            ["/home/ubuntu/tfrecords/tfrecords22"], num_epochs=None)
        _, data_serialized = tf.TFRecordReader().read(filename_queue_two)

        feature = tf.parse_single_example(data_serialized,
                                          features={
                                              'label': tf.FixedLenFeature([1], dtype=tf.int64),
                                              'index': tf.VarLenFeature(dtype=tf.int64),
                                              'value': tf.VarLenFeature(dtype=tf.float32),
                                          })

        feature_dense = tf.sparse_to_dense(sparse_indices=tf.sparse_tensor_to_dense(features['index']),
                                           output_shape=[33762578, ],
                                           sparse_values=tf.sparse_tensor_to_dense(features['value']))

        y_hat = tf.reduce_sum(tf.mul(weight, feature_dense))

        test_error = tf.mul(tf.cast(
            tf.sub(tf.constant([1], dtype=tf.int64),
                   tf.mul(tf.cast(tf.sign(y_hat), tf.int64), features['label'])),
            tf.float32), 0.5)

    # For every VM run 10000 iterations and afer every 100 iteration calculate the test error using 2000 data points
    # Collect the errors in a file called async_errors
    with tf.Session("grpc://vm-22-%d:2222" % (FLAGS.task_index + 1)) as sess:
        if FLAGS.task_index == 0:
            sess.run(tf.initialize_all_variables())
        tf.train.start_queue_runners(sess=sess)
        for i in range(0, 10000):
            if FLAGS.task_index == 0:
                run_cross_validation = i != 0 and i % 100 == 0
                if run_cross_validation:
                    total_error_rate = 0
                    for j in range(0, 2000):
                        sess.run(test_error)
                        total_error_rate += test_error.eval()[0]
                    total_error_rate = (total_error_rate / 2000) * 100
                    with open("async_errors", "a+") as aynsc_errors:
                        aynsc_errors.write(
                            "\nIteration- " + str(i) + " Error Rate Percentage: " + str(total_error_rate))
            sess.run(assign_op)
            print(weight.eval())
        sess.close()
