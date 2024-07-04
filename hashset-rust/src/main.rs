use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub struct SimpleLinkedList<T: PartialEq> {
    head: Option<Box<Node<T>>>,
}

// data should always be present, next node is Optional
struct Node<T: PartialEq> {
    data: T,
    next: Option<Box<Node<T>>>,
}

impl<T: PartialEq> SimpleLinkedList<T> {
    pub fn new() -> Self {
        Self { head: None }
    }
    pub fn is_empty(&self) -> bool {
        self.head.is_none()
    }
    pub fn len(&self) -> usize {
        let mut size = 0usize;
        let mut curr = &self.head;
        while let Some(x) = curr {
            size += 1;
            curr = &x.next;
        }
        size
    }

    pub fn contains(&self, element: &T) -> bool {
        let mut curr = &self.head;
        while let Some(x) = curr {
            if &x.data == element {
                return true;
            } else {
                curr = &x.next;
            }
        }
        false
    }

    pub fn push(&mut self, element: T) {
        // Transfer the owenership if previous head to this node
        // We must use `.take()` method else implement Copy trait
        // Taking ownership makes sense here because if the head node
        // is dropped, all consecutive nodes should be dropped as well
        let node = Box::new(Node {
            data: element,
            next: self.head.take(),
        });
        self.head = Some(node);
    }
    pub fn pop(&mut self) -> Option<T> {
        // Same thing about ownership here
        // `.take()`: Takes the value out of the option, leaving a None in its place.
        if let Some(node) = self.head.take() {
            self.head = node.next;
            return Some(node.data);
        }
        None
    }
    pub fn peek(&self) -> Option<&T> {
        // [[wiki]]
        // Maps an Option<T> to Option<U> by applying a function
        // to a contained value (if Some) or returns None (if None)
        self.head.as_ref().map(|node| &node.data)
    }
    #[must_use]
    pub fn rev(self) -> SimpleLinkedList<T> {
        let mut list = Self::new();
        let mut curr = self.head;
        while let Some(node) = curr {
            list.push(node.data);
            curr = node.next;
        }
        list
    }
}

struct HashSet<T: PartialEq> {
    array: Vec<SimpleLinkedList<T>>,
    capacity: usize,
}

fn get_hash<T: Hash>(t: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    t.hash(&mut hasher);
    hasher.finish()
}

impl<T: Hash + PartialEq> HashSet<T> {
    pub fn new(capacity: usize) -> Self {
        let mut array: Vec<SimpleLinkedList<T>> = vec![];
        (0..capacity).for_each(|_| array.push(SimpleLinkedList::new()));
        Self { array, capacity }
    }

    pub fn insert(&mut self, item: T) {
        let idx = get_hash(&item) as usize % self.capacity();
        if !self.array[idx].contains(&item) {
            self.array[idx].push(item);
        }
    }

    #[allow(dead_code)]
    pub fn contains(&self, item: &T) -> bool {
        let idx = get_hash(item) as usize % self.capacity();
        self.array[idx].contains(item)
    }

    pub fn len(&self) -> usize {
        self.array.iter().map(|ll| ll.len()).sum()
    }

    pub fn capacity(&self) -> &usize {
        &self.capacity
    }
}

fn main() {
    let mut set = HashSet::<i32>::new(16);
    let data = [1, 3, 7, 9, 2, 3, 4, 6, 7, 8, 43, 4, 2, 1, 8, 3, 4, 6, 7, 9];
    data.iter().for_each(|item| set.insert(item.clone()));
    assert_eq!(set.len(), 9);
    // let std_set = std::collections::HashSet::from(data);
    // println!("{}", std_set.len());
}
