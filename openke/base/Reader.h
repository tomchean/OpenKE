#ifndef READER_H
#define READER_H
#include "Setting.h"
#include "Triple.h"
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <cmath>

// freqEnt means entity appear times
// freqRel means relation appear times
INT *freqRel, *freqEnt;

// lefHead means entity start index sort by head
// rigHead means entity end index sort by head
// lefTail means entity start index sort by tail
// rigTail means entity end index sort by tail
// lefRel means entity start index sort by relation 
// rigRel means entity end index sort by relation
INT *lefHead, *rigHead;
INT *lefTail, *rigTail;
INT *lefRel, *rigRel;

// left_mean means relation count / #(have relation in entity and entity is head)
// right_mean means relation count / #(have relation in entity and entity is tail)
REAL *left_mean, *right_mean;

REAL *prob;

Triple *trainList;
Triple *trainHead;
Triple *trainTail;
Triple *trainRel;


INT *slefHead, *srigHead;
INT *slefTail, *srigTail;
INT *slefRel, *srigRel;
REAL *sleft_mean, *sright_mean;

Triple *streamList;
Triple *streamHead;
Triple *streamTail;
Triple *streamRel;

INT *testLef, *testRig;
INT *validLef, *validRig;

extern "C"
void importProb(REAL temp){
    if (prob != NULL)
        free(prob);
    FILE *fin;
    fin = fopen((inPath + "kl_prob.txt").c_str(), "r");
    printf("Current temperature:%f\n", temp);
    prob = (REAL *)calloc(relationTotal * (relationTotal - 1), sizeof(REAL));
    INT tmp;
    for (INT i = 0; i < relationTotal * (relationTotal - 1); ++i){
        tmp = fscanf(fin, "%f", &prob[i]);
    }
    REAL sum = 0.0;
    for (INT i = 0; i < relationTotal; ++i) {
        for (INT j = 0; j < relationTotal-1; ++j){
            REAL tmp = exp(-prob[i * (relationTotal - 1) + j] / temp);
            sum += tmp;
            prob[i * (relationTotal - 1) + j] = tmp;
        }
        for (INT j = 0; j < relationTotal-1; ++j){
            prob[i*(relationTotal-1)+j] /= sum;
        }
        sum = 0;
    }
    fclose(fin);
}

extern "C"
void importStreamFiles() {

	printf("The toolkit is importing datasets.\n");
	FILE *fin;
	int tmp;

    if (train_file == "")
        fin = fopen((inPath + "stream2id.txt").c_str(), "r");
    else
        fin = fopen(train_file.c_str(), "r");
	tmp = fscanf(fin, "%ld", &streamTotal);
	streamList = (Triple *)calloc(streamTotal, sizeof(Triple));
	streamHead = (Triple *)calloc(streamTotal, sizeof(Triple));
	streamTail = (Triple *)calloc(streamTotal, sizeof(Triple));
	streamRel = (Triple *)calloc(streamTotal, sizeof(Triple));

	for (INT i = 0; i < streamTotal; i++) {
		tmp = fscanf(fin, "%ld", &streamList[i].h);
		tmp = fscanf(fin, "%ld", &streamList[i].t);
		tmp = fscanf(fin, "%ld", &streamList[i].r);
	}
	fclose(fin);


	streamHead[0] = streamTail[0] = streamRel[0] = streamList[0];
	freqEnt[streamList[0].t] += 1;
	freqEnt[streamList[0].h] += 1;
	freqRel[streamList[0].r] += 1;

	for (INT i = 1; i < streamTotal; i++) {
		streamHead[i] = streamTail[i] = streamRel[i] = streamList[i];
		freqEnt[streamList[i].t]++;
		freqEnt[streamList[i].h]++;
		freqRel[streamList[i].r]++;

    }

	std::sort(streamHead, streamHead + streamTotal, Triple::cmp_head);
	std::sort(streamTail, streamTail + streamTotal, Triple::cmp_tail);
	std::sort(streamRel, streamRel + streamTotal, Triple::cmp_rel);
	printf("The total of stream triples is %ld.\n", streamTotal);

	slefHead = (INT *)calloc(entityTotal, sizeof(INT));
	srigHead = (INT *)calloc(entityTotal, sizeof(INT));
	slefTail = (INT *)calloc(entityTotal, sizeof(INT));
	srigTail = (INT *)calloc(entityTotal, sizeof(INT));
	slefRel = (INT *)calloc(entityTotal, sizeof(INT));
	srigRel = (INT *)calloc(entityTotal, sizeof(INT));

	memset(srigHead, -1, sizeof(INT)*entityTotal);
	memset(srigTail, -1, sizeof(INT)*entityTotal);
	memset(srigRel, -1, sizeof(INT)*entityTotal);

	for (INT i = 1; i < streamTotal; i++) {
		if (streamTail[i].t != streamTail[i - 1].t) {
			srigTail[streamTail[i - 1].t] = i - 1;
			slefTail[streamTail[i].t] = i;
		}
		if (streamHead[i].h != streamHead[i - 1].h) {
			srigHead[streamHead[i - 1].h] = i - 1;
			slefHead[streamHead[i].h] = i;
		}
		if (streamRel[i].h != streamRel[i - 1].h) {
			srigRel[streamRel[i - 1].h] = i - 1;
			slefRel[streamRel[i].h] = i;
		}
	}

	slefHead[streamHead[0].h] = 0;
	srigHead[streamHead[streamTotal - 1].h] = streamTotal - 1;
	slefTail[streamTail[0].t] = 0;
	srigTail[streamTail[streamTotal - 1].t] = streamTotal - 1;
	slefRel[streamRel[0].h] = 0;
	srigRel[streamRel[streamTotal - 1].h] = streamTotal - 1;

	sleft_mean = (REAL *)calloc(relationTotal,sizeof(REAL));
	sright_mean = (REAL *)calloc(relationTotal,sizeof(REAL));

	for (INT i = 0; i < entityTotal; i++) {
		for (INT j = slefHead[i] + 1; j <= srigHead[i]; j++)
			if (streamHead[j].r != streamHead[j - 1].r)
				sleft_mean[streamHead[j].r] += 1.0;
		if (slefHead[i] <= srigHead[i])
			sleft_mean[streamHead[slefHead[i]].r] += 1.0;
		for (INT j = slefTail[i] + 1; j <= srigTail[i]; j++)
			if (streamTail[j].r != streamTail[j - 1].r)
				sright_mean[streamTail[j].r] += 1.0;
		if (slefTail[i] <= srigTail[i])
			sright_mean[streamTail[slefTail[i]].r] += 1.0;
	}
	for (INT i = 0; i < relationTotal; i++) {
		sleft_mean[i] = freqRel[i] / sleft_mean[i];
		sright_mean[i] = freqRel[i] / sright_mean[i];
	}
}

extern "C"
void importTrainFiles() {

	printf("The toolkit is importing datasets.\n");
	FILE *fin;
	int tmp;

    if (rel_file == "")
	    fin = fopen((inPath + "relation2id.txt").c_str(), "r");
    else
        fin = fopen(rel_file.c_str(), "r");
	tmp = fscanf(fin, "%ld", &relationTotal);
	printf("The total of relations is %ld.\n", relationTotal);
	fclose(fin);

    if (ent_file == "")
        fin = fopen((inPath + "entity2id.txt").c_str(), "r");
    else
        fin = fopen(ent_file.c_str(), "r");
	tmp = fscanf(fin, "%ld", &entityTotal);
	printf("The total of entities is %ld.\n", entityTotal);
	fclose(fin);

    if (train_file == "")
        fin = fopen((inPath + "train2id.txt").c_str(), "r");
    else
        fin = fopen(train_file.c_str(), "r");
	tmp = fscanf(fin, "%ld", &trainTotal);
	trainList = (Triple *)calloc(trainTotal, sizeof(Triple));
	trainHead = (Triple *)calloc(trainTotal, sizeof(Triple));
	trainTail = (Triple *)calloc(trainTotal, sizeof(Triple));
	trainRel = (Triple *)calloc(trainTotal, sizeof(Triple));
	freqRel = (INT *)calloc(relationTotal, sizeof(INT));
	freqEnt = (INT *)calloc(entityTotal, sizeof(INT));
	for (INT i = 0; i < trainTotal; i++) {
		tmp = fscanf(fin, "%ld", &trainList[i].h);
		tmp = fscanf(fin, "%ld", &trainList[i].t);
		tmp = fscanf(fin, "%ld", &trainList[i].r);
	}
	fclose(fin);
	std::sort(trainList, trainList + trainTotal, Triple::cmp_head);
	tmp = trainTotal; trainTotal = 1;
	trainHead[0] = trainTail[0] = trainRel[0] = trainList[0];
	freqEnt[trainList[0].t] += 1;
	freqEnt[trainList[0].h] += 1;
	freqRel[trainList[0].r] += 1;
	for (INT i = 1; i < tmp; i++)
		if (trainList[i].h != trainList[i - 1].h || trainList[i].r != trainList[i - 1].r || trainList[i].t != trainList[i - 1].t) {
			trainHead[trainTotal] = trainTail[trainTotal] = trainRel[trainTotal] = trainList[trainTotal] = trainList[i];
			trainTotal++;
			freqEnt[trainList[i].t]++;
			freqEnt[trainList[i].h]++;
			freqRel[trainList[i].r]++;
		}

	std::sort(trainHead, trainHead + trainTotal, Triple::cmp_head);
	std::sort(trainTail, trainTail + trainTotal, Triple::cmp_tail);
	std::sort(trainRel, trainRel + trainTotal, Triple::cmp_rel);
	printf("The total of train triples is %ld.\n", trainTotal);

	lefHead = (INT *)calloc(entityTotal, sizeof(INT));
	rigHead = (INT *)calloc(entityTotal, sizeof(INT));
	lefTail = (INT *)calloc(entityTotal, sizeof(INT));
	rigTail = (INT *)calloc(entityTotal, sizeof(INT));
	lefRel = (INT *)calloc(entityTotal, sizeof(INT));
	rigRel = (INT *)calloc(entityTotal, sizeof(INT));
	memset(rigHead, -1, sizeof(INT)*entityTotal);
	memset(rigTail, -1, sizeof(INT)*entityTotal);
	memset(rigRel, -1, sizeof(INT)*entityTotal);
	for (INT i = 1; i < trainTotal; i++) {
		if (trainTail[i].t != trainTail[i - 1].t) {
			rigTail[trainTail[i - 1].t] = i - 1;
			lefTail[trainTail[i].t] = i;
		}
		if (trainHead[i].h != trainHead[i - 1].h) {
			rigHead[trainHead[i - 1].h] = i - 1;
			lefHead[trainHead[i].h] = i;
		}
		if (trainRel[i].h != trainRel[i - 1].h) {
			rigRel[trainRel[i - 1].h] = i - 1;
			lefRel[trainRel[i].h] = i;
		}
	}
	lefHead[trainHead[0].h] = 0;
	rigHead[trainHead[trainTotal - 1].h] = trainTotal - 1;
	lefTail[trainTail[0].t] = 0;
	rigTail[trainTail[trainTotal - 1].t] = trainTotal - 1;
	lefRel[trainRel[0].h] = 0;
	rigRel[trainRel[trainTotal - 1].h] = trainTotal - 1;

	left_mean = (REAL *)calloc(relationTotal,sizeof(REAL));
	right_mean = (REAL *)calloc(relationTotal,sizeof(REAL));
	for (INT i = 0; i < entityTotal; i++) {
		for (INT j = lefHead[i] + 1; j <= rigHead[i]; j++)
			if (trainHead[j].r != trainHead[j - 1].r)
				left_mean[trainHead[j].r] += 1.0;
		if (lefHead[i] <= rigHead[i])
			left_mean[trainHead[lefHead[i]].r] += 1.0;
		for (INT j = lefTail[i] + 1; j <= rigTail[i]; j++)
			if (trainTail[j].r != trainTail[j - 1].r)
				right_mean[trainTail[j].r] += 1.0;
		if (lefTail[i] <= rigTail[i])
			right_mean[trainTail[lefTail[i]].r] += 1.0;
	}
	for (INT i = 0; i < relationTotal; i++) {
		left_mean[i] = freqRel[i] / left_mean[i];
		right_mean[i] = freqRel[i] / right_mean[i];
	}
}

Triple *testList;
Triple *validList;
Triple *tripleList;

extern "C"
void importTestFiles() {
    FILE *fin;
    INT tmp;
    
    if (rel_file == "")
	    fin = fopen((inPath + "relation2id.txt").c_str(), "r");
    else
        fin = fopen(rel_file.c_str(), "r");
    tmp = fscanf(fin, "%ld", &relationTotal);
    fclose(fin);

    if (ent_file == "")
        fin = fopen((inPath + "entity2id.txt").c_str(), "r");
    else
        fin = fopen(ent_file.c_str(), "r");
    tmp = fscanf(fin, "%ld", &entityTotal);
    fclose(fin);

    FILE* f_kb1, * f_kb2, * f_kb3;
    if (train_file == "")
        f_kb2 = fopen((inPath + "train2id.txt").c_str(), "r");
    else
        f_kb2 = fopen(train_file.c_str(), "r");
    if (test_file == "")
        f_kb1 = fopen((inPath + "test2id.txt").c_str(), "r");
    else
        f_kb1 = fopen(test_file.c_str(), "r");
    if (valid_file == "")
        f_kb3 = fopen((inPath + "valid2id.txt").c_str(), "r");
    else
        f_kb3 = fopen(valid_file.c_str(), "r");
    tmp = fscanf(f_kb1, "%ld", &testTotal);
    tmp = fscanf(f_kb2, "%ld", &trainTotal);
    tmp = fscanf(f_kb3, "%ld", &validTotal);
    tripleTotal = testTotal + trainTotal + validTotal;
    testList = (Triple *)calloc(testTotal, sizeof(Triple));
    validList = (Triple *)calloc(validTotal, sizeof(Triple));
    tripleList = (Triple *)calloc(tripleTotal, sizeof(Triple));
    for (INT i = 0; i < testTotal; i++) {
        tmp = fscanf(f_kb1, "%ld", &testList[i].h);
        tmp = fscanf(f_kb1, "%ld", &testList[i].t);
        tmp = fscanf(f_kb1, "%ld", &testList[i].r);
        tripleList[i] = testList[i];
    }
    for (INT i = 0; i < trainTotal; i++) {
        tmp = fscanf(f_kb2, "%ld", &tripleList[i + testTotal].h);
        tmp = fscanf(f_kb2, "%ld", &tripleList[i + testTotal].t);
        tmp = fscanf(f_kb2, "%ld", &tripleList[i + testTotal].r);
    }
    for (INT i = 0; i < validTotal; i++) {
        tmp = fscanf(f_kb3, "%ld", &tripleList[i + testTotal + trainTotal].h);
        tmp = fscanf(f_kb3, "%ld", &tripleList[i + testTotal + trainTotal].t);
        tmp = fscanf(f_kb3, "%ld", &tripleList[i + testTotal + trainTotal].r);
        validList[i] = tripleList[i + testTotal + trainTotal];
    }
    fclose(f_kb1);
    fclose(f_kb2);
    fclose(f_kb3);

    std::sort(tripleList, tripleList + tripleTotal, Triple::cmp_head);
    std::sort(testList, testList + testTotal, Triple::cmp_rel2);
    std::sort(validList, validList + validTotal, Triple::cmp_rel2);
    printf("The total of test triples is %ld.\n", testTotal);
    printf("The total of valid triples is %ld.\n", validTotal);

    testLef = (INT *)calloc(relationTotal, sizeof(INT));
    testRig = (INT *)calloc(relationTotal, sizeof(INT));
    memset(testLef, -1, sizeof(INT) * relationTotal);
    memset(testRig, -1, sizeof(INT) * relationTotal);
    for (INT i = 1; i < testTotal; i++) {
	if (testList[i].r != testList[i-1].r) {
	    testRig[testList[i-1].r] = i - 1;
	    testLef[testList[i].r] = i;
	}
    }
    testLef[testList[0].r] = 0;
    testRig[testList[testTotal - 1].r] = testTotal - 1;

    validLef = (INT *)calloc(relationTotal, sizeof(INT));
    validRig = (INT *)calloc(relationTotal, sizeof(INT));
    memset(validLef, -1, sizeof(INT)*relationTotal);
    memset(validRig, -1, sizeof(INT)*relationTotal);
    for (INT i = 1; i < validTotal; i++) {
	if (validList[i].r != validList[i-1].r) {
	    validRig[validList[i-1].r] = i - 1;
	    validLef[validList[i].r] = i;
	}
    }
    validLef[validList[0].r] = 0;
    validRig[validList[validTotal - 1].r] = validTotal - 1;
}

INT* head_lef;
INT* head_rig;
INT* tail_lef;
INT* tail_rig;
INT* head_type;
INT* tail_type;

extern "C"
void importTypeFiles() {

    head_lef = (INT *)calloc(relationTotal, sizeof(INT));
    head_rig = (INT *)calloc(relationTotal, sizeof(INT));
    tail_lef = (INT *)calloc(relationTotal, sizeof(INT));
    tail_rig = (INT *)calloc(relationTotal, sizeof(INT));
    INT total_lef = 0;
    INT total_rig = 0;
    FILE* f_type = fopen((inPath + "type_constrain.txt").c_str(),"r");
    INT tmp;
    tmp = fscanf(f_type, "%ld", &tmp);
    for (INT i = 0; i < relationTotal; i++) {
        INT rel, tot;
        tmp = fscanf(f_type, "%ld %ld", &rel, &tot);
        for (INT j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%ld", &tmp);
            total_lef++;
        }
        tmp = fscanf(f_type, "%ld%ld", &rel, &tot);
        for (INT j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%ld", &tmp);
            total_rig++;
        }
    }
    fclose(f_type);
    head_type = (INT *)calloc(total_lef, sizeof(INT)); 
    tail_type = (INT *)calloc(total_rig, sizeof(INT));
    total_lef = 0;
    total_rig = 0;
    f_type = fopen((inPath + "type_constrain.txt").c_str(),"r");
    tmp = fscanf(f_type, "%ld", &tmp);
    for (INT i = 0; i < relationTotal; i++) {
        INT rel, tot;
        tmp = fscanf(f_type, "%ld%ld", &rel, &tot);
        head_lef[rel] = total_lef;
        for (INT j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%ld", &head_type[total_lef]);
            total_lef++;
        }
        head_rig[rel] = total_lef;
        std::sort(head_type + head_lef[rel], head_type + head_rig[rel]);
        tmp = fscanf(f_type, "%ld%ld", &rel, &tot);
        tail_lef[rel] = total_rig;
        for (INT j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%ld", &tail_type[total_rig]);
            total_rig++;
        }
        tail_rig[rel] = total_rig;
        std::sort(tail_type + tail_lef[rel], tail_type + tail_rig[rel]);
    }
    fclose(f_type);
}


#endif
