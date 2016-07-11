/**
 *  \file parallel-mergesort.cc
 *
 *  \brief Implement your parallel mergesort in this file.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <omp.h>
#include "sort.hh"

/**
 * Searches for an index Q in 2nd array where
 * all elements with index < Q are less than mid element
 * of first array
 */
int splitPoint(keytype* &arr, int first, int last, int mid1) {
	int ans = 0;
	while (last > first) {
		ans = (first + last) / 2;
		if (arr[ans] >= arr[mid1]) {
			last = ans;
		} else {
			first = ans + 1;
		}
	}
	return last;
}

/**
 * parallel merge which returns the result in auxiliary array
 * auxiliary array (aux) is shared between different recursions,
 * so merge functions uses the space corresponding to the main array
 * indicated by auxOffset as starting index
 * and the length determined by the sum sizes of both sub arrays.
 * s1, e1 are start and end of sub array1. s2, e2 are start and end of sub array2.
 * sz1 and sz2 are sizes of sub array1 and sub array2 respectively.
 */
void pMerge(keytype* &arr, keytype* &aux, int auxOffset, int s1, int e1, int s2,
		int e2, int depth) {

	int sz1 = e1 - s1 + 1, sz2 = e2 - s2 + 1;

	/*use recursion if there are more than 2 elements in both arrays
	 * and number of threads available is greater than 1
	 * else fall back to sequential(non-recursive merge)
	 */
	if (sz1 > 2 && sz2 > 2 && depth > 1) {

		int m1 = (s1 + e1) / 2;
		int m2 = splitPoint(arr, s2, e2 + 1, m1);
		int qOffset = 0;

		if (m2 == e2 + 1) {
			/*all elements of array 2 are less than mid element of array 1*/
			qOffset = auxOffset + (m1 - s1) + (e2 - s2 + 1);
			memcpy(aux + qOffset, arr + m1, (e1 - m1 + 1) * sizeof(keytype));
			pMerge(arr, aux, auxOffset, s1, m1 - 1, s2, e2, depth);
			return;
		} else if (m2 == s2) {
			/*all elements of array2 are greater than mid element of array 1*/
			memcpy(aux + auxOffset, arr + s1, (m1 - s1 + 1) * sizeof(keytype));
			pMerge(arr, aux, auxOffset + (m1 - s1 + 1), m1 + 1, e1, s2, e2,
					depth);
			return;
		} else {
			qOffset = auxOffset + (m1 - s1) + (m2 - s2);
			aux[qOffset] = arr[m1];
#pragma omp task shared(arr, aux, depth)
			{
				pMerge(arr, aux, auxOffset, s1, m1 - 1, s2, m2 - 1, depth / 2);
			}
			pMerge(arr, aux, qOffset + 1, m1 + 1, e1, m2, e2, depth / 2);
#pragma omp taskwait
		}
	} else {
		/*sequential merge*/
		int itr1 = s1, itr2 = s2, limit = auxOffset + sz1 + sz2;
		for (int i = auxOffset; i < limit; i++) {
			if (itr1 > e1) {
				memcpy(aux + i, arr + itr2, (limit - i) * sizeof(keytype));
				break;
			} else if (itr2 > e2) {
				memcpy(aux + i, arr + itr1, (limit - i) * sizeof(keytype));
				break;
			} else if (arr[itr1] < arr[itr2]) {
				aux[i] = arr[itr1];
				itr1++;
			} else {
				aux[i] = arr[itr2];
				itr2++;
			}
		}
	}
	return;
}

/*Parallel Merge sort*/
void mergeSort(keytype* &arr, keytype* &aux, int start, int end, int depth) {

	if (start >= end)
		return;

	int mid = start + ((end - start) / 2);
	/*spawn new threads for recursion
	 * if available threads are greater than 1
	 */
	if (depth > 1) {
#pragma omp task shared(arr, aux)
		{
			mergeSort(arr, aux, start, mid, depth / 2);
		}
		mergeSort(arr, aux, mid + 1, end, depth / 2);
#pragma omp taskwait
	} else {
		mergeSort(arr, aux, start, mid, depth);
		mergeSort(arr, aux, mid + 1, end, depth);
	}

	pMerge(arr, aux, start, start, mid, mid + 1, end, depth);
	memcpy(arr + start, aux + start, (end - start + 1) * sizeof(keytype));

	return;
}

void
parallelSort (int N, keytype* A)
{
	/*depth is used to avoid spawning more threads than available in system*/
	int depth = 0;

	/*Auxiliary array for extra space needed during merge*/
	keytype *aux = (keytype *) malloc(N * sizeof(keytype));

#pragma omp parallel
	{
#pragma omp single nowait
		{
			/*get the number of threads in the system*/
			depth = omp_get_num_threads();
			mergeSort(A, aux, 0, N - 1, depth);
		}
	}

	free(aux);

}

/* eof */
