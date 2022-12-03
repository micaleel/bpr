#include <stdio.h>
#include <stdlib.h>


typedef struct {
	double value;
	int idx;
} my_index ;

int cmpfunc(const void * a, const void * b){
	my_index *indexA = (my_index *) a;
	my_index * indexB = (my_index *) b;
	if (indexA->value < indexB->value) return -1;
	if (indexB->value < indexA->value) return 1;
	return 0;
}

void argsort_row(double * A, int * out,my_index * row_index, int K){
	int i;

	for (i=0; i<K; i++){
		my_index m;
		m.value=A[i];
		m.idx=i;
		row_index[i] = m;
	}		

	qsort(row_index, K, sizeof(my_index), cmpfunc);

	for (i=0; i<K; i++){
		out[i] = row_index[i].idx;
	}
}

int binary_search_row(double *arr, int *P, int n, double target) { 
    // Corner cases 
    if (target <= arr[0]) 
        return 0; 
    if (target >= arr[n - 1]) 
        return n-1; 
  
    // Doing binary search 
    int i = 0, j = n, mid = 0; 
    while (i < j) { 
        mid = (i + j) / 2; 
  
        if (arr[mid] == target){
            while (P[mid] == 0)
				mid -= 1;
			return mid; 
		}

        /* If target is less than array element, 
            then search in left */
        if (target < arr[mid]) { 
  
            // If target is greater than previous 
            // to mid, return closest of two 
            if (mid > 0 && target > arr[mid - 1]) {
				while (P[mid]==0){
					mid-=1;
				}
				return mid;
			}
  
            /* Repeat for left half */
            j = mid; 
        } 
  
        // If target is greater than mid 
        else { 
            if (mid < n - 1 && target < arr[mid + 1]) {
				mid=mid+1;
				while (P[mid]==0)
					mid-=1;
				return mid;
			}
                
            // update i 
            i = mid + 1;  
        } 
    } 
  
    // Only single element left after search 
    return mid; 
}