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

fn mysnd<A,B>((x,y): (A,B)) -> B{
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
struct Lam<V,B> {var : PhantomData<V>, body : PhantomData<B>}
type t4 = (bool,bool);
type KV<K,V> = (K,V);

trait Lookup<K> {
    type Output;
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
#[derive(Clone, Debug)]
enum LamExpr {
    Lam( Box<LamExpr>),
    App( Box<LamExpr>,  Box<LamExpr>),
    Var( usize),
    Lit(i64) // Pair, Fst, Snd
}
use LamExpr::*;

/*
split enum out may
struct Lam<U>( )

or... 
reflection into typeclass via 
class Reflect s
    yada yada
    yada yada
    yada yada
use macro to make fresh reflection class every time.

PhantomData<True>
trait Reflect where
    type Output;
    val : Output;
impl Reflect for True where
    type Output = bool;
    val = true;
impl Reflect for False{
    type Output = bool;
    val = false;
    }
-- reflecting ordinary types
-- some kind of dyanmic typing might pull this off?
-- Any
impl Reflect for Vec<U> where
    type Output = VecR<U::Output>; 
    val = VecR(val);

reflection can perform something very singleton like.
typeclasses cna perform higher rank functionality.

(Reflect fi) (Reflect fo) (Reflect fs) => reifyWith x match x => 

reifyWith ::  MyGADT s' -> (Reflect s, Reflect fs) => s -> fs) (Reflect s) (Reflect s) -> f s'
reifyWith 

reifyWith 
refiyWithVecMyGADT Reflect s => MyGADT s' -> Vec<s::out> -> Vec<s::out> -> Vec<s> -> Vec<s::out>
matchish Reflect s => 

reifyWith GADT s -> (Reflect s s' => f s' -> f s') -> f s
Reflect<Int> s

Reify
Eq<Int> for U{

}

Eq<>

typclass equality

Eq<U> {
    cast :: Self -> U
    uncast :: U -> Self
}

Eq<i64> for i64{
      ?leibnitz?
      cast = id
      uncast = id
}

s : Eq<Int>
s' : Eq<Bool>

(Type -> GARY) -> 

refl : Eq<A,A>
withRefl 


App<Vec<Type>,A>
impl Eval for App<Vec<Type>,A> where A : Eval{
    type Output = Vec<A::Output>
}
impl Eval for i64{
    type Output = i64;
}

impl Abstract for Vec<A> {
    type Ouput = App<Vec<Type>,A::Abstract
}

impl EvalOrAbstract<A> {

}

Unify f<Int> (App<Vec<Type>,>) where
 -> App<F,Int> -> App<F,Bool> -> 
 reify( c1 : impl Eval<App<F,Int>>, c2 : impl ) where fi = App<F,Int> as Eval :: Output 
refiy( c1 : fi, )

App

we might need something more like the coq case syntax giving the explanation on how to build the type of the result.
This is only acceptable with a tactic language.

desctrcut<F> :: GADT s -> Eval App<F,Int> -> Eval App<F,Bool> -> Eval App<F,s>
destructEq :: Eq a b -> Eval App2<F,A,A> -> Eval App2<F,A,B>

struct Lam<A,B>
type MyExpr<A> = Lam<A, Vec<A>> // sort of a typelevel HOAS
lam!(fred = \a b -> b) -> type Fred0<A,B> = Fred<>
need to curry everythign? 
lam!(fred = \a b -> Vec a)   type Fred<A,B> = Vec<A>
pat!(fred a b = )
lam!(fred = \a b c -> ) ==> type Fred<A,B,C> = 

impl Apply Fred<Type,Type,Type>




Abtract 
HKT = Abstract

fn<U>( )

f s -> f s -> f s ->


R -> R -> R
   U ->  U -> U -> U
require GADT destructor for every pair of GADT to destruct and context to put it in?
destruct
maybe via reflection? reflect type variables into 
desugar into lambda   |x y z| =>  
a higher order macro. Can we do that?

smart constructors
fn lam(a : LamExpr<U>) -> LamExpr<Arr<U,V>>

returning a type that only implements a trait is a way of exitentializing.

*/
// https://gist.github.com/kyleheadley/af149f7452c8163667e2b047743a9e44
// typevel lambda calc
fn eval(mut env : Vec<LamExpr>, term : LamExpr) -> LamExpr {

    match term{
        LamExpr::App(l, v) => {
            let env2 = env.clone();
            let v2 = eval( env2, *v);
            env.push(v2);
            eval( env  , *l)
            },
        LamExpr::Lam(b) => eval(env, *b),
        LamExpr::Var(i) => {
            if env.len() > i{
                env[i].clone()
            }
            else {
                Var(i)
            }            
        },
        Lit(x) => Lit(x)
    }
}

fn app(f : LamExpr, x : LamExpr) -> LamExpr{
    LamExpr::App(Box::new(f), Box::new(x))
}
fn lam(b : LamExpr) -> LamExpr{
    Lam(Box::new(b))
}

fn id() -> LamExpr {
   LamExpr::Lam(Box::new(Var(0)))
}

fn pair(a : LamExpr, b : LamExpr) -> LamExpr{
    lam(app(app(Var(0), a), b))
}
fn fst() -> LamExpr{
    lam(lam( Var(1)))
}

fn snd() -> LamExpr{
    lam(lam( Var(0)))
}

fn main() {

    println!("Hello, world!");
    dbg!(MINE);
    dbg!(mysnd((1,"Hi")));


    dbg!(eval(Vec::new(),  app(id(),Lit(1))  ));
    dbg!(eval(Vec::new(),  id()));
    dbg!(eval(Vec::new(),  app(fst(), pair(Lit(0),Lit(1)))     ));

}
