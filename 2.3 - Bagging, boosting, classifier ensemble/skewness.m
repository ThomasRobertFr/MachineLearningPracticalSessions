function retval = skewness (x, dim)

  if (nargin ~= 1 && nargin ~= 2)
    print_usage ();
  end

  if (~ (isnumeric (x) || islogical (x)))
    error ('skewness: X must be a numeric vector or matrix');
  end

  nd = ndims (x);
  sz = size (x);
  if (nargin ~= 2)
    % Find the first non-singleton dimension.
    dim = find (sz > 1, 1);
    if(isempty(dim))
        dim = 1;
    end
  else
    if (~(isscalar (dim) && dim == fix (dim)) || ~(1 <= dim && dim <= nd))
      error ('skewness: DIM must be an integer and a valid dimension');
    end
  end

  n = sz(dim);
  sz(dim) = 1;
  x = center (x, dim);  % center also promotes integer to double for next line
  retval = zeros (sz, class (x));
  s = std (x, [], dim);
  idx = find (s > 0);
  x = sum (x .^ 3, dim);
  retval(idx) = x(idx) ./ (n * s(idx) .^ 3);

end


%!assert(skewness ([-1,0,1]), 0);
%!assert(skewness ([-2,0,1]) < 0);
%!assert(skewness ([-1,0,2]) > 0);
%!assert(skewness ([-3,0,1]) == -1*skewness([-1,0,3]));
%!test
%! x = [0; 0; 0; 1];
%! y = [x, 2*x];
%! assert(all (abs (skewness (y) - [0.75, 0.75]) < sqrt (eps)));

%!assert (skewness (single(1)), single(0));

%% Test input validation
%!error skewness ()
%!error skewness (1, 2, 3)
%!error skewness (['A'; 'B'])
%!error skewness (1, ones(2,2))
%!error skewness (1, 1.5)
%!error skewness (1, 0)
%!error skewness (1, 3)
