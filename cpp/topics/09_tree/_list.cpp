
#include "_list.h"

list_header_t* list_header_new()
{
    list_header_t* lt = (list_header_t*) MM_ALLOC(sizeof(list_header_t));
    BASIC_ASSERT(lt != NULL);
    lt->node_num = 0;
    LLIST_HEAD_RESET(lt);
    return lt;
}

void list_header_init(list_header_t *p_header)
{
    LLIST_HEAD_RESET(p_header);
    p_header->node_num = 0;
}

void list_clear(list_header_t *p_header)
{
    if (LLIST_IS_EMPTY(p_header))
        return;

    do {
        LList_Entry_t *prev;
        LList_Entry_t *curr;
        LLIST_WHILE_START(p_header, curr, LList_Entry_t) {
            prev = curr;
            LLIST_WHILE_NEXT(curr, LList_Entry_t);
            MM_FREE(prev);
        }
        LLIST_HEAD_RESET(p_header);
    } while(0);

    p_header->node_num = 0;

    return;
}

void list_delete(list_header_t *p_header)
{
    list_clear(p_header);
    MM_FREE(p_header);
    return;
}

int list_append(list_header_t *head, void *node)
{
    LLIST_INSERT_LAST(head, node);
    head->node_num++;
    return 0;
}

void *list_pop(list_header_t *hd)
{
    void *last;
    BASIC_ASSERT(hd != NULL);

    if (LLIST_IS_EMPTY(hd)) {
        return NULL;
    } else {
        last = LLIST_LAST(hd);
        DLLIST_REMOVE_LAST_SAFELY(hd);
        hd->node_num--;
    }

    return last;
}

void* list_at(list_header_t *hd, int32_t idx)
{
    BASIC_ASSERT(hd != NULL);

    int32_t new_idx = idx >= 0 ? idx : hd->node_num+idx;
    BASIC_ASSERT(new_idx >= 0);
    BASIC_ASSERT((size_t)new_idx < hd->node_num);

    LList_Entry_t *curr;
    int32_t i=0;

    LLIST_FOREACH(hd, curr, LList_Entry_t) {
        if (i == new_idx) {
            return curr;
        }
        i++;
    }

    return NULL;
}

void *list_remove(list_header_t *hd, int32_t idx)
{
    BASIC_ASSERT(hd != NULL);
    
    int32_t new_idx = idx >= 0 ? idx : hd->node_num+idx;
    BASIC_ASSERT(new_idx >= 0);
    BASIC_ASSERT((size_t)new_idx < hd->node_num);

    int32_t i = 0;
    LList_Entry_t *curr;

    if (LLIST_IS_EMPTY(hd)) {
        return NULL;
    } else {
        LLIST_FOREACH(hd, curr, LList_Entry_t) {
            if (i == new_idx) {
                LLIST_REMOVE_NODE_SAFELY(hd, curr);
                hd->node_num--;
                return curr;
            }
            i++;
        }
    }
    BASIC_ASSERT(0);
    return NULL;
}

void list_diagnose(list_header_t *hd CALLER_PARA3)
{
	// Length check
    void *temp;
	uint32_t count = 0;
	list_header_t *currNode = (list_header_t *)LLIST_FIRST(hd);

    if (LLIST_IS_EMPTY(hd)) {
        return;
    }

	while (currNode != NULL) {
		count++;
		currNode = (list_header_t *)LLIST_NEXT(currNode);
	}
	if (count != hd->node_num) {
		DUMPD(count);
		DUMPD(hd->node_num);
		CALLER_ASSERT3(0);
	}
	// Length check in backward
	count = 0;
    temp = LLIST_LAST(hd);
	currNode = (list_header_t *)temp;
	while (currNode != hd) {
		count++;
		currNode = (list_header_t *)LLIST_PREV(currNode);
	}
	CALLER_ASSERT3(count == hd->node_num);

	// Generic checkPRLOC
	if (LLIST_IS_NOT_EMPTY(hd)) {
		CALLER_ASSERT3(LLIST_PREV(LLIST_FIRST(hd)) == hd);
		CALLER_ASSERT3(LLIST_TAIL(hd) != NULL);
		CALLER_ASSERT3(LLIST_NEXT(LLIST_TAIL(hd)) == NULL);

		// Check relation address between 2 nodes
		for (list_header_t *currNode = (list_header_t *)LLIST_HEAD(hd); currNode != NULL; currNode = (list_header_t *)LLIST_NEXT(currNode)) {
			if (LLIST_NEXT(currNode) != NULL) {
				CALLER_ASSERT3(currNode == (list_header_t *)LLIST_PREV(LLIST_NEXT(currNode)));
			}
		}
        temp = LLIST_LAST(hd);
		for (list_header_t *currNode = (list_header_t *)temp; currNode != hd; currNode = (list_header_t *)LLIST_PREV(currNode)) {
			if (LLIST_PREV(currNode) != NULL) {
				CALLER_ASSERT3(currNode == (list_header_t *)LLIST_NEXT(LLIST_PREV(currNode)));
			}
		}
	} else {
		CALLER_ASSERT3(LLIST_TAIL(hd) == NULL);
	}
}

typedef struct {
    LList_Entry_t en;
    int data;
} diagnose_demo_t;

void LibLinkedList_Diagnose_Demo(void)
{
    list_header_t hd_inst;
    list_header_t *hd = &hd_inst;
    list_header_init(hd);

	diagnose_demo_t a, b, c;
	a.data = 111;
	b.data = 222;
	c.data = 333;

    list_append(hd, &a);
    LIST_DIAGNOSE(hd);
    list_append(hd, &b);
    LIST_DIAGNOSE(hd);
    list_append(hd, &c);
    LIST_DIAGNOSE(hd);

    diagnose_demo_t *curr = (diagnose_demo_t *)list_at(hd, 1);
    LIST_DIAGNOSE(hd);
    BASIC_ASSERT(curr->data == 222);
	DUMPND(curr->data);

    curr = (diagnose_demo_t *)list_remove(hd, 1);
    LIST_DIAGNOSE(hd);
    DUMPND(curr->data);
    
    curr = (diagnose_demo_t *)list_pop(hd);
    LIST_DIAGNOSE(hd);
    BASIC_ASSERT(curr->data == 333);
    DUMPND(curr->data);

    curr = (diagnose_demo_t *)list_pop(hd);
    LIST_DIAGNOSE(hd);
    BASIC_ASSERT(curr->data == 111);
    DUMPND(curr->data);
}