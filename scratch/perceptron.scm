(define train-X '((0 0 1) (0 1 1) (1 0 1) (1 1 1)))
(define train-y '(1 1 0 1))
(define weights '(0.0 0.0 0.0))

(define (dot X weights)
  (define (dot-aux X weights result)
    (cond
      ((or (null? X) (null? weights)) result)
      (else (+ (* (car X) (car weights)) (dot-aux (cdr X) (cdr weights) result)))))
  (dot-aux X weights 0.0))

(define (step x)
  (if (> x 1.0) 1.0 0.0))

(define (forward X weights)
  (step (dot X weights)))

(define (predict inputs weights)
  (cond
    ((or (null? inputs) (null? weights)) 
     (newline))
    (else
      (display (format "input: ~a, predict: ~a\n" (car inputs) (forward (car inputs) weights)))
      (predict (cdr inputs) weights))))

(define (add-values x y)
  (define (add-values-aux x y result)
    (cond
      ((or (null? x) (null? y)) (reverse result))
      (else (add-values-aux (cdr x) (cdr y) (cons (+ (car x) (car y)) result)))))
  (add-values-aux x y '()))

(define (backward X loss epsilon)
  (define (backward-aux X diff)
    (cond
      ((null? X) (reverse diff))
      (else 
        (backward-aux (cdr X) (cons (* loss (* epsilon (car X))) diff)))))
  (set! weights (add-values weights (backward-aux X '()))))

(predict train-X weights)

(define (train Xs ys epsilon)
  (define (train-aux X y epsilon)
    (backward X (- y (forward X weights)) epsilon))
  (cond
    ((and (null? Xs) (null? ys)) 
     (display (format "weights: ~a\n" weights)))
    (else 
      (train-aux (car Xs) (car ys) epsilon)
      (train (cdr Xs) (cdr ys) epsilon))))

(define (fit Xs ys epsilon epoch)
  (cond
    ((zero? epoch) (newline))
    (else
      (display (format "Progress [~a] " epoch))
      (train Xs ys epsilon)
      (fit Xs ys epsilon (sub1 epoch)))))



(train train-X train-y 0.1)
(fit train-X train-y 0.01 100)
(predict train-X weights)
