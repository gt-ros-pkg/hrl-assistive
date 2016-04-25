

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
