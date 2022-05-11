from model import FFDNet

if __name__ == "__main__":

    # initialize FFDNet model
    ffd_net = FFDNet()

    ffd_net.dncnn.summary()
    ffd_net.compile(optimizer='adam', loss='mse', metrics=['mse'])

    EPOCHS=5
    #training
    batch_sz = 100
    import time
    start_time=time.time()
    ffd_net.fit(X_train, downsample(Y_train),
                    epochs=EPOCHS,
                    verbose=1,
                    shuffle=True,
                    validation_data=(downsample(X_valid), downsample(Y_valid)),
                    batch_size=batch_sz,
                )
    print("--- %s Training time in minutes ---" % ((time.time() - start_time)/60))


    import math;
    #test
    scores = ffd_net.evaluate(X_test, downsample(Y_test), batch_size=80, verbose=1)
    scores
    print('\nTest MSE dB: %.5f loss: %.5f' % (10*math.log(scores[0], 10),scores[1]))
