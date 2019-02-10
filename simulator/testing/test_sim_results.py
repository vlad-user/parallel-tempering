from simulator.summary_extractor import SummaryExtractor

def test_replica_noise_values(simulation_name, simulation_num=0):
    
  se = SummaryExtractor(simulation_name)
  n_replicas = se.get_description()['n_replicas']
  for r in range(n_replicas):
    x, y = se.get_summary('noise_values',
                          replica_id=r,
                          simulation_num=simulation_num)

    noise_list = se.get_description()['noise_list']

    assert len(list(x)) == len(list(y))
    prev_y = y[0]

    # check correct transitions
    for i in range(1, len(list(y))):
      try:
        predicate = (prev_y == y[i]
                     or abs(noise_list.index(prev_y) - noise_list.index(y[i])) == 1)
        assert predicate
      except AssertionError as exc:
        err_msg = ('previous noise val:'
                   + str(prev_y)
                   + ', current noise val:'
                   + str(y[i])
                   + '. But index of '
                   + str(prev_y)
                   + ' is '
                   + str(noise_list.index(prev_y))
                   + ' and index of '
                   + str(y[i])
                   + ' is '
                   + str(noise_list.index(y[i])))

        raise AssertionError(err_msg) from exc
      prev_y = y[i]

    # check that there is no two replicas with the same noise value
    # at the same time

    reps = {r:se.get_summary('noise_values', r, simulation_num)[1]
            for r in range(n_replicas)}

    for i in range(len(y)):
      curr_vals = list(set([k[i] for k in reps.values()]))

      assert len(curr_vals) == n_replicas

