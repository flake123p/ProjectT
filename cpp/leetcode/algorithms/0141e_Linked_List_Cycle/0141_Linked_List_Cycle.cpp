#include <cstddef>
#include <iostream>

/** Leetcode 141**/
/** Definition for singly-linked list. **/
struct ListNode	{
		int val;
		ListNode *next;
		ListNode(int x): val(x), next(NULL)	{}
};

bool hasCycle(ListNode* head){
	ListNode* fast = head;
	ListNode* slow = head;
	while (fast){
		if(!fast->next){
			return false;
		}
		fast = fast -> next -> next;
		slow = slow -> next;
		if(slow==fast){
			return true;
		}
	}
	return false;
}

int main() {
	std::cout << "Here is the given sequence 1, 2, 3, 4, 5, which is not cycle" << "\n";
	ListNode ln1(1);
	ListNode ln2(2);
	ListNode ln3(3);
	ListNode ln4(4);
	ListNode ln5(5);
	ln1.next = &ln2;
	ln2.next = &ln3;
	ln3.next = &ln4;
	ln4.next = &ln5;
	ln5.next = nullptr;
	std::cout << ln1.next << "\n";
	std::cout << &ln2 << "\n";
	std::cout << ln2.next << "\n";
	std::cout << &ln3 << "\n";
	std::cout << ln3.next << "\n";
	std::cout << &ln4 << "\n";
	std::cout << ln4.next << "\n";
	std::cout << &ln5 << "\n";
	std::cout << ln5.next << "\n";
	std::cout <<  "The seq is not cycle:" << hasCycle(&ln1) << "\n";
	std::cout << "Here is the given sequence 6, 7, 8, 9, 6, which is  cycle" << "\n";
	ListNode ln6(6);
	ListNode ln7(7);
	ListNode ln8(8);
	ListNode ln9(9);
	ln6.next = &ln7;
	ln7.next = &ln8;
	ln8.next = &ln9;
	ln9.next = &ln6;
	std::cout << ln9.next << "\n";
	std::cout << &ln6 << "\n";
	std::cout <<  "The seq is cycle:" << hasCycle(&ln6) << "\n";
}

