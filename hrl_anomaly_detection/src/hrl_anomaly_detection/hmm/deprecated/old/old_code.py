

    def learn_likelihoods_progress_par(self, i, n, m, A, B, pi, X_train, g_mu, g_sig):
        l_likelihood_mean = 0.0
        l_likelihood_mean2 = 0.0
        l_statePosterior = np.zeros(self.nState)

        for j in xrange(n):
            results = Parallel(n_jobs=-1)(delayed(computeLikelihood)(self.F, k, X_train[j][:k*self.nEmissionDim], g_mu, g_sig, self.nEmissionDim, A, B, pi) for k in xrange(1, m))

            g_post = np.sum([r[0] for r in results], axis=0)
            g_lhood, g_lhood2, prop_sum = np.sum([r[1:] for r in results], axis=0)

            l_statePosterior += g_post / prop_sum / float(n)
            l_likelihood_mean += g_lhood / prop_sum / float(n)
            l_likelihood_mean2 += g_lhood2 / prop_sum / float(n)

        return i, l_statePosterior, l_likelihood_mean, np.sqrt(l_likelihood_mean2 - l_likelihood_mean**2)


    def likelihoods(self, X):
        X_test = self.convert_sequence(X, emission=False)

        final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0].tolist())
        logp = self.ml.loglikelihood(final_ts_obj)
        post = np.array(self.ml.posterior(final_ts_obj))

        n = len(np.squeeze(X[0]))

        # Find the best posterior distribution
        min_index, min_dist = self.findBestPosteriorDistribution(post[n-1])

        ll_likelihood = logp
        ll_state_idx  = min_index
        ll_likelihood_mu  = self.ll_mu[min_index]
        ll_likelihood_std = self.ll_std[min_index] #self.ll_mu[min_index] + ths_mult*self.ll_std[min_index]

        return ll_likelihood, ll_state_idx, ll_likelihood_mu, ll_likelihood_std

    def allLikelihoods(self, X):
        X_test = self.convert_sequence(X, emission=False)

        m = len(np.squeeze(X[0]))

        ll_likelihood = np.zeros(m)
        ll_state_idx  = np.zeros(m)
        ll_likelihood_mu  = np.zeros(m)
        ll_likelihood_std = np.zeros(m)
        for i in xrange(1, m):
            final_ts_obj = ghmm.EmissionSequence(self.F, X_test[0,:i*self.nEmissionDim].tolist())
            logp = self.ml.loglikelihood(final_ts_obj)
            post = np.array(self.ml.posterior(final_ts_obj))

            # Find the best posterior distribution
            min_index, min_dist = self.findBestPosteriorDistribution(post[i-1])

            ll_likelihood[i] = logp
            ll_state_idx[i]  = min_index
            ll_likelihood_mu[i]  = self.ll_mu[min_index]
            ll_likelihood_std[i] = self.ll_std[min_index] #self.ll_mu[min_index] + ths_mult*self.ll_std[min_index]

        return ll_likelihood, ll_state_idx, ll_likelihood_mu, ll_likelihood_std


    def partial_fit(self, xData, nTrain, scale, weight=8.0):
        '''
        data: dimension x sample x length
        '''
        A  = copy.copy(self.A)
        B  = copy.copy(self.B)
        pi = copy.copy(self.pi)

        new_B = copy.copy(self.B)

        t_features = []
        mus        = []
        covs       = []
        for i in xrange(self.nState):
            t_features.append( B[i][0] + [ float(i) / float(self.nState)*scale/2.0 ])
            mus.append( B[i][0] )
            covs.append( B[i][1] )
        t_features = np.array(t_features)
        mus        = np.array(mus)
        covs       = np.array(covs)

        # update b ------------------------------------------------------------
        # mu
        x_l = [[] for i in xrange(self.nState)]
        X   = np.swapaxes(xData, 0, 1) # sample x dim x length
        seq_len = len(X[0][0])
        for i in xrange(len(X)):
            sample = np.swapaxes(X[i], 0, 1) # length x dim

            idx_l = []
            for j in xrange(len(sample)):
                feature = np.array( sample[j].tolist() + [float(j)/float(len(sample))*scale/2.0 ] )

                min_dist = 10000
                min_idx  = 0
                for idx, t_feature in enumerate(t_features):
                    dist = np.linalg.norm(t_feature-feature)
                    if dist < min_dist:
                        min_dist = dist
                        min_idx  = idx

                x_l[min_idx].append(feature[:-1].tolist())

        for i in xrange(len(mus)):
            if len(x_l[i]) > 0:
                avg_x = np.mean(x_l[i], axis=0)
                new_B[i][0] = list( ( float(nTrain-1)*mus[i] + avg_x*weight ) / float(nTrain+(weight-1) ) ) # specialized for single input


        # Normalize the state prior and transition values.
        A_sum = np.sum(A, axis=1)
        for i in xrange(self.nState):
            A[i,:] /= A_sum[i]
        pi /= np.sum(pi)

        # Daehyung: What is the shape and type of input data?
        xData = [np.array(data) for data in xData]
        X_ptrain = util.convert_sequence(xData) # Training input
        X_ptrain = np.squeeze(X_ptrain)

        final_ts_obj = ghmm.EmissionSequence(self.F, X_ptrain.tolist())        
        (alpha, scale) = self.ml.forward(final_ts_obj)
        beta = self.ml.backward(final_ts_obj, scale)

        ## print np.shape(alpha), np.shape(beta), type(alpha), type(beta)

        est_A = np.zeros((self.nState, self.nState))
        new_A = np.zeros((self.nState, self.nState))
        for i in xrange(self.nState):
            for j in xrange(self.nState):

                temp1 = 0.0
                temp2 = 0.0
                for t in xrange(seq_len):
                    p = multivariate_normal.pdf( X[0][:,t], mean=mus[j], \
                                                 cov=np.reshape(covs[j], \
                                                                (self.nEmissionDim, self.nEmissionDim)))
                    temp1 += alpha[t-1][i] * A[i,j] * p * beta[t][j]
                    temp2 += alpha[t-1][i] * beta[t][j]

                if temp1 == 0.0 or temp2 == 0.0: est_A[i,j] = 0
                else: est_A[i,j] = temp1/temp2
                    
                new_A[i,j] = (float(nTrain-len(xData))*A[i,j] + est_A[i,j]*weight) / float(nTrain + (weight-1.0) )

        # Normalize the state prior and transition values.
        A_sum = np.sum(new_A, axis=1)
        for i in xrange(self.nState):
            new_A[i,:] /= A_sum[i]
        pi /= np.sum(pi)
            
        self.set_hmm_object(new_A, new_B, pi)
        return new_A, new_B, pi


    ## def predict(self, X):
    ##     '''
    ##     ???????????????????
    ##     HMM is just a generative model. What will be prediction result?
    ##     Which code is using this fuction?
    ##     '''
    ##     X = np.squeeze(X)
    ##     X_test = X.tolist()

    ##     mu_l = np.zeros(self.nEmissionDim)
    ##     cov_l = np.zeros(self.nEmissionDim**2)

    ##     if self.verbose: print self.F
    ##     final_ts_obj = ghmm.EmissionSequence(self.F, X_test) # is it neccessary?

    ##     try:
    ##         # alpha: X_test length y # latent States at the moment t when state i is ended
    ##         # test_profile_length x number_of_hidden_state
    ##         (alpha, scale) = self.ml.forward(final_ts_obj)
    ##         alpha = np.array(alpha)
    ##     except:
    ##         if self.verbose: print "No alpha is available !!"
            
    ##     f = lambda x: round(x, 12)
    ##     for i in range(len(alpha)):
    ##         alpha[i] = map(f, alpha[i])
    ##     alpha[-1] = map(f, alpha[-1])
        
    ##     n = len(X_test)
    ##     pred_numerator = 0.0

    ##     for j in xrange(self.nState): # N+1
    ##         total = np.sum(self.A[:,j]*alpha[n/self.nEmissionDim-1,:]) #* scaling_factor
    ##         [mus, covars] = self.B[j]

    ##         ## print mu1, mu2, cov11, cov12, cov21, cov22, total
    ##         pred_numerator += total

    ##         for i in xrange(mu_l.size):
    ##             mu_l[i] += mus[i]*total
    ##         for i in xrange(cov_l.size):
    ##             cov_l[i] += covars[i] * (total**2)

    ##     return mu_l, cov_l
