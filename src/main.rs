// use std::option;
use std::marker::PhantomData;
use std::ops::Add;
const MINE : Option<i64> = None;
// https://github.com/gtestault/primitive-recursive-functions

struct Z {}
struct S<T>(PhantomData<T>);
type T1 = S<Z>;
struct True {}
struct False {}

trait Not{
    type Res;
}
impl Not for True {
    type Res = False;
}

impl Not for False {
    type Res = True;
}
trait Nand<T> {
    type Res;
}
/*
impl Nand<False> for True{
    type Res = True;
}*/
impl<T> Nand<T> for True where T : Not {
    type Res = <T as Not>::Res;
} 

//let y : () = 

// HKT
//https://gist.github.com/CMCDragonkai/a5638f50c87d49f815b8
// Interesint. a HKT is somoething that can have interior exchanged.

// Equality is similar...
/*
trait Eq<A>  { // requires HKT
    type T; // context can change
    fn HKT<U,C> 

}


*/


struct OptionTag {}

trait Taggable {
    type Tag;
}

impl<T> Taggable for Option<T>{
    type Tag = OptionTag;
}

fn myid<A>(x : A) -> A {
    x
} 

fn snd<A,B>((x,y): (A,B)) -> B{
    y
}

struct V2<T> {
    x : T,
    y : T
}

impl<T> Add for V2<T> where T : Add<Output = T> {
    type Output = V2<T>;
    fn add(self, other : V2<T>) -> V2<T> {
        V2 { 
            x : self.x + other.x,
            y : self.y + other.y}
    }
}

// Make a compile time lisp.
// PrimOps have to be traits
// Make a macro to turn legit lisp expressions ()
// quote
// use typelevel num 
// using refl package, one could make a gcase macro
// to make casing on gadts easier.
struct Cons<A,B> {h: PhantomData<A>, t: PhantomData<B> }
struct Nil {}
type t3 = Cons<Z , Nil>;
// struct Apply<A,B> {f: PhantomData<A>, x: PhantomData<B> }
// struct Lam<V,B> {var : PhantomData<V>, body : PhantomData<B>}
type t4 = (bool,bool);
type KV<K,V> = (K,V);

trait Lookup<N> {
    type T;
}

impl<A,B> Lookup<Z> for Cons<A,B>{
    type T = A;
}

impl<N,A,B> Lookup<S<N>> for Cons<A,B> where B : Lookup<N> {
    type T = B::T;
}


//impl<K> Lookup<K> for (( , ), )
// kind of the same thing as HKT
/*
trait Lam<V, A> {
    type Output;
}
struct Id {};

impl<V,A> Lam<V,A> for Id {
    type Output = A;
}
*/
/*
struct Id {}
struct Snd {}
struct Fst {}
trait Apply<X> {
    type Output;
}
impl<X> Apply<X> for Id{
    type Output = X;
}

impl<X,Y> Apply<(X,Y)> for Snd{
    type Output = Y;
}

impl<X,Y> Apply<(X,Y)> for Fst{
    type Output = X;
}
*/
struct Id<X>(PhantomData<X>);
struct Fst<X>(PhantomData<X>);
trait Apply {
    type Output;
}

impl<X> Apply for Id<X> {
    type Output = X;
}

impl<X,Y> Apply for Fst<(X,Y)> {
    type Output = X;
}
// trait Eval // trait Apply?

struct TypeTag {}

struct List<A> {
    head : A, 
    tail : Box<List<A>>
}

trait ApplyHKT<U> {
    type Output;
}

trait AbstractHKT {
    type Output;
}
// splitting the suggested HKT typeclass into two method.
// may want tag types the be macro generated to be units.
impl<U> AbstractHKT for List<U>{
    type Output = List<TypeTag>;
}

impl<U> ApplyHKT<U> for List<TypeTag>{
    type Output = List<U>;
}
enum Lit<U> {
    LitInt(i64, PhantomData<U>),
    LitBool(bool, PhantomData<U>)
}
/*
trait GadtTest : HKT<i64> HKT<bool> {
    fn litint(i64) -> Self::HKT<i64>::T;
    fn litbool(bool) -> Self::HKT<bool>::T;
}
*/

struct Lam<B> (PhantomData<B>);

type T5 = Lam<List<Z>>; // The de bruijn indexed List type

trait ApplyDB<Env> {
    type T;
}

trait Eval<Env> {

}

struct Apply2<L,X> {lam : PhantomData<L> , var : PhantomData<X>  }

impl<Env> Eval where {

} 
impl<B, Env> ApplyDB<Env> Lam<B> where Apply<Cons<Arg, Env>>  {

}


// sized Vec?
// I mean I think rust has sized arrayc built in, but still.

fn main() {
    println!("Hello, world!");
    dbg!(MINE);
    dbg!(snd((1,"Hi")));

}
