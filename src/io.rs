//! 入出力に関するモジュールです.
use std::any::type_name;
use std::io::{BufRead, Stdin, StdinLock};
use std::mem::MaybeUninit;
use std::str::FromStr;
use std::sync::Once;

/// `BufRead`からトークンを読み出すための構造体です.
///
/// トークンは空白文字(スペース,改行,行頭復帰,タブ)で区切られるものとします.
/// またトークンを読むためにはEOFまたは改行(`\n`)が与えられる必要があります.
pub struct Scanner<R: BufRead> {
    reader: R,
    buf: Vec<u8>,
    pos_in_buf: usize,
    token_num: usize,
}

impl<R: BufRead> Scanner<R> {
    /// `reader`からトークンを読み出す`Scanner`を返します.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            buf: vec![],
            pos_in_buf: 0,
            token_num: 0,
        }
    }

    /// 次のトークンを$T$型に変換して返します.
    #[allow(clippy::should_implement_trait)]
    pub fn next<T: FromStr>(&mut self) -> T {
        let res = self.next_token();
        match res.parse() {
            Ok(v) => v,
            _ => panic!(
                "Failed to parse token {} as {}",
                self.token_num,
                type_name::<T>()
            ),
        }
    }

    /// 次のトークンをスキップします.
    pub fn skip(&mut self) {
        self.next_token();
    }

    fn next_token(&mut self) -> &str {
        self.advance_to_next_token();
        let mut end_pos = self.pos_in_buf;
        while end_pos < self.buf.len() && !Self::is_whitespace(self.buf[end_pos]) {
            end_pos += 1;
        }

        let res = unsafe { std::str::from_utf8_unchecked(&self.buf[self.pos_in_buf..end_pos]) };
        self.pos_in_buf = end_pos;
        self.token_num += 1;
        res
    }

    fn advance_to_next_token(&mut self) {
        loop {
            self.skip_whitespace();
            if self.pos_in_buf < self.buf.len() {
                break;
            }
            self.read_to_buf();
            self.pos_in_buf = 0;
        }
    }

    fn skip_whitespace(&mut self) {
        while self.pos_in_buf < self.buf.len() && Self::is_whitespace(self.buf[self.pos_in_buf]) {
            self.pos_in_buf += 1;
        }
    }

    fn is_whitespace(byte: u8) -> bool {
        byte == b' ' || byte == b'\r' || byte == b'\n' || byte == b'\t'
    }
    fn read_to_buf(&mut self) {
        self.buf.clear();
        let read_bytes = self.reader.read_until(b'\n', &mut self.buf).unwrap();
        if read_bytes == 0 {
            panic!("Unexpected EOF");
        }
    }
}
/// 標準入力をソースとする`Scanner`です.
pub fn stdin_scanner() -> &'static mut Scanner<StdinLock<'static>> {
    static mut STDIN: MaybeUninit<Stdin> = MaybeUninit::uninit();
    static mut STDIN_SCANNER: MaybeUninit<Scanner<StdinLock<'_>>> = MaybeUninit::uninit();
    static ONCE: Once = Once::new();

    unsafe {
        ONCE.call_once(|| {
            STDIN = MaybeUninit::new(std::io::stdin());
            let lock = (*STDIN.as_ptr()).lock();
            STDIN_SCANNER = MaybeUninit::new(Scanner::new(lock));
        });

        &mut *STDIN_SCANNER.as_mut_ptr()
    }
}

#[macro_export]
#[doc(hidden)]
macro_rules! read_value {
    // vector
    ($scanner:ident,[$type:tt; $len:expr]) => {
        (0..$len)
            .map(|_| $crate::read_value!($scanner, $type))
            .collect::<Vec<_>>()
    };
    // string
    ($scanner:ident,chars) => {
        $scanner.next::<String>().chars().collect::<Vec<_>>()
    };
    // tuple
    ($scanner:ident,($($type:tt),*)) => {
        ($($crate::read_value!($scanner, $type)),*)
    };
    // otherwise
    ($scanner:ident,$type:ty) => {
        $scanner.next::<$type>()
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! scan_inner {
    // base case
    ($scanner:ident;) => {};
    ($scanner:expr;,) => {};
    // immutable
    ($scanner:ident; $var:ident : $type:tt, $($rest:tt)*) => {
        let $var = $crate::read_value!($scanner,$type);
        $crate::scan_inner!($scanner; $($rest)*)
    };
    // mutable
    ($scanner:ident; mut $var:ident : $type:tt, $($rest:tt)*) => {
        #[allow(unused_mut)]
        let mut $var = $crate::read_value!($scanner,$type);
        $crate::scan_inner!($scanner; $($rest)*)
    };
}

#[macro_export]
macro_rules! scan {
    ($scanner:expr; $($tt:tt)*) => {
        #[allow(unused_variables)]
        let temp = &mut $scanner;
        $crate::scan_inner!(temp; $($tt)*,)
    };
    ($($tt:tt)*) => {
        #[allow(unused_variables)]
        let scanner = $crate::io::stdin_scanner();
        $crate::scan_inner!(scanner; $($tt)*,)
    };
}
pub use scan;

#[macro_export]
macro_rules! print_immediate {
    ($($arg:tt)*) => {
        let out = $crate::stdout_lock();
        write!(out, $($arg)*).unwrap();
        out.flush().unwrap();
    };
}
pub use print_immediate;

#[macro_export]
macro_rules! println_immediate {
    ($($arg:tt)*) => {
        let out = $crate::stdout_lock();
        writeln!(out, $($arg)*).unwrap();
        out.flush().unwrap();
    };
}
pub use println_immediate;

#[cfg(test)]
mod test {
    use super::{scan, Scanner};

    #[test]
    fn scanner_basic() {
        let s = "a bb  ccc\n\ndddd\t\n\teeeee\rffffff\n\t\r ggggggg   ";
        let mut sc = Scanner::new(s.as_bytes());

        let actual: Vec<_> = (0..7).map(|_| sc.next::<String>()).collect();

        let expected: Vec<String> = vec!["a", "bb", "ccc", "dddd", "eeeee", "ffffff", "ggggggg"]
            .into_iter()
            .map(|s| s.into())
            .collect();

        assert_eq!(expected, actual);
    }

    #[test]
    fn scan_macro() {
        let s = "12 abc 3456 788.123 hello,world 9999 12 34 56 78 90 1 foobar";
        let mut sc = Scanner::new(s.as_bytes());

        scan! {
            sc; mut a:usize, b:String, c:i32, d:f64
        }

        a *= 2;
        assert_eq!(a, 24);
        assert_eq!(b, String::from("abc"));
        assert_eq!(c, 3456);
        assert!((d - 788.123).abs() < std::f64::EPSILON);

        scan! {
            sc; mut a: String, mut b: f64
        }

        assert_eq!(a, String::from("hello,world"));
        assert!((b - 9999.).abs() < std::f64::EPSILON);

        scan! {
            sc; l: [u32; 4], mut r: [isize; 2], cs: chars
        }
        r[0] *= 2;

        assert_eq!(l, vec![12, 34, 56, 78]);
        assert_eq!(r, vec![180, 1]);
        assert_eq!(cs, "foobar".chars().collect::<Vec<_>>());
    }

    #[test]
    fn scan_tuple() {
        let s = "12 aa 34 56 bb 78 90 cc 123";
        let mut sc = Scanner::new(s.as_bytes());

        scan! {
            sc; actual: [(usize,String,i32); 3]
        }

        let expected = vec![
            (12, "aa".to_owned(), 34),
            (56, "bb".to_owned(), 78),
            (90, "cc".to_owned(), 123),
        ];

        assert_eq!(actual, expected);
    }

    #[test]
    fn scan_macro_nest() {
        let s = "1 2 3\n4 5 6\n7 8 9\n10 11 12";

        scan! {
            Scanner::new(s.as_bytes()); mat: [[usize; 3]; 4]
        }

        assert_eq!(
            mat,
            vec![
                vec![1, 2, 3],
                vec![4, 5, 6],
                vec![7, 8, 9],
                vec![10, 11, 12]
            ]
        );
    }

    struct DummyStruct<R: std::io::BufRead> {
        sc: Scanner<R>,
    }

    #[test]
    fn semantics() {
        scan! {
            Scanner::new("".as_bytes());
        }
        let mut sc = Scanner::new("".as_bytes());
        scan! {
            sc;
        }
        let mut s = DummyStruct { sc };
        scan! {
            s.sc;
        }

        scan! {}
    }
}
